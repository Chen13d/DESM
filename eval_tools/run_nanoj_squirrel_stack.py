#!/usr/bin/env python3
"""
Batch runner for NanoJ-SQUIRREL (No GPU) in a local Fiji/ImageJ install.

The input stack is interpreted in groups of five slices:
  Input, Output_channel_1, Output_channel_2, Reference_channel_1, Reference_channel_2

For each group, the script runs:
  Output_channel_1 vs Reference_channel_1
  Output_channel_2 vs Reference_channel_2

It writes a summary CSV with RSP/RSE and saves the input/reference/output,
RSF-convolved output, and RSE error-map images by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import stat
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DESM_ROOT = SCRIPT_DIR.parent
DEFAULT_IMAGEJ_DIR = Path(os.environ.get("FIJI_HOME", "Fiji.app"))
DEFAULT_STACK = (
    DESM_ROOT
    / "visualization"
    / "Experimental_data_eval"
    / "Micro_Mito"
    / "test"
    / "225_0.040.tif"
)


JYTHON_RUNNER = r'''
from __future__ import print_function

import json
import os
import traceback

from ij import IJ, ImagePlus, ImageStack, WindowManager
from ij.measure import ResultsTable
from java.lang import System, NoSuchFieldException
from openCLfree.squirrel.java.gui import ErrorMapUNIQORNNoCL_

CONFIG = json.loads(r"""__CONFIG_JSON__""")


def write_text(path, text):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "w") as f:
        f.write(text)


def append_text(path, text):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "a") as f:
        f.write(text)


def csv_escape(value):
    if value is None:
        value = ""
    value = str(value)
    if '"' in value:
        value = value.replace('"', '""')
    if "," in value or "\n" in value or '"' in value:
        value = '"' + value + '"'
    return value


def find_field(obj, name):
    cls = obj.getClass()
    while cls is not None:
        try:
            field = cls.getDeclaredField(name)
            field.setAccessible(True)
            return field
        except NoSuchFieldException:
            cls = cls.getSuperclass()
    raise AttributeError(name)


def set_obj(obj, name, value):
    find_field(obj, name).set(obj, value)


def set_int(obj, name, value):
    find_field(obj, name).setInt(obj, int(value))


def set_bool(obj, name, value):
    find_field(obj, name).setBoolean(obj, bool(value))


def make_slice_image(source, slice_index, title):
    fp = source.getStack().getProcessor(slice_index).convertToFloatProcessor()
    stack = ImageStack(fp.getWidth(), fp.getHeight())
    stack.addSlice(title, fp.duplicate())
    imp = ImagePlus(title, stack)
    cal = source.getCalibration().copy()
    if CONFIG["pixel_size_nm"] > 0:
        cal.pixelWidth = CONFIG["pixel_size_nm"]
        cal.pixelHeight = CONFIG["pixel_size_nm"]
        cal.pixelDepth = CONFIG["pixel_size_nm"]
        cal.setUnit("nm")
    imp.setCalibration(cal)
    return imp


def close_all_images():
    ids = WindowManager.getIDList()
    if ids is not None:
        for image_id in ids:
            imp = WindowManager.getImage(image_id)
            if imp is not None:
                imp.changes = False
                imp.close()
    frame = WindowManager.getFrame("RSP and RSE values")
    if frame is not None:
        frame.dispose()


def prepare_plugin(ref_imp, sr_imp, ref_title, sr_title):
    sr_stack = sr_imp.getImageStack().convertToFloat()
    ref_stack = ref_imp.getImageStack().convertToFloat()

    w_sr = sr_stack.getWidth()
    h_sr = sr_stack.getHeight()
    w_ref = ref_stack.getWidth()
    h_ref = ref_stack.getHeight()
    if w_ref == 0 or h_ref == 0:
        raise RuntimeError("Reference image has zero width or height")
    if w_sr % w_ref != 0 or h_sr % h_ref != 0:
        raise RuntimeError("SR image size must be an integer multiple of reference image size")
    mag_x = w_sr // w_ref
    mag_y = h_sr // h_ref
    if mag_x != mag_y:
        raise RuntimeError("X/Y magnification mismatch: %s vs %s" % (mag_x, mag_y))

    plugin = ErrorMapUNIQORNNoCL_()
    set_obj(plugin, "titleRefImage", ref_title)
    set_obj(plugin, "titleSRImage", sr_title)
    set_obj(plugin, "impRef", ref_imp)
    set_obj(plugin, "impSR", sr_imp)
    set_obj(plugin, "imsRef", ref_stack)
    set_obj(plugin, "imsSR", sr_stack)
    set_int(plugin, "w_SR", w_sr)
    set_int(plugin, "h_SR", h_sr)
    set_int(plugin, "magnification", mag_x)
    set_int(plugin, "magnification2", mag_x * mag_x)
    set_int(plugin, "nSlicesSR", sr_imp.getStackSize())
    set_int(plugin, "nSlicesRef", ref_imp.getStackSize())

    set_bool(plugin, "smartBoundary", CONFIG["smart_boundary"])
    set_bool(plugin, "framePurge", CONFIG["frame_purge"])
    set_bool(plugin, "borderControl", CONFIG["border_control"])
    set_bool(plugin, "doRegistration", CONFIG["do_registration"])
    set_int(plugin, "maxExpectedMisalignment", CONFIG["max_expected_misalignment"])
    set_bool(plugin, "showIntensityNormalised", False)
    set_bool(plugin, "showConvolved", CONFIG["save_image_outputs"])
    set_bool(plugin, "showPositiveNegative", CONFIG["show_positive_negative"])
    return plugin


def read_results_table():
    rt = ResultsTable.getResultsTable("RSP and RSE values")
    if rt is None:
        rt = ResultsTable.getActiveTable()
    if rt is None or rt.size() == 0:
        raise RuntimeError(
            "SQUIRREL results table was not created. Run without java.awt.headless=true."
        )
    row = rt.size() - 1
    rsp = rt.getValue("RSP (Resolution Scaled Pearson-Correlation)", row)
    rse = rt.getValue("RSE (Resolution Scaled Error)", row)
    overblur = rt.getStringValue("Overblur warning", row)
    return rsp, rse, overblur


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def save_image(imp, group_index, filename):
    out_dir = os.path.join(CONFIG["output_dir"], "saved_images", "group_%03d" % group_index)
    ensure_dir(out_dir)
    path = os.path.join(out_dir, filename)
    IJ.saveAsTiff(imp, path)
    return path


def run_pair(source, group_index, channel_index):
    base = (group_index - 1) * 5
    input_slice = base + 1
    if channel_index == 1:
        sr_slice = base + 2
        ref_slice = base + 4
    else:
        sr_slice = base + 3
        ref_slice = base + 5

    input_title = "group_%03d_input" % group_index
    sr_title = "group_%03d_channel_%d_output" % (group_index, channel_index)
    ref_title = "group_%03d_channel_%d_reference" % (group_index, channel_index)
    input_imp = make_slice_image(source, input_slice, input_title)
    sr_imp = make_slice_image(source, sr_slice, sr_title)
    ref_imp = make_slice_image(source, ref_slice, ref_title)

    plugin = prepare_plugin(ref_imp, sr_imp, ref_title, sr_title)
    plugin.execute()
    rsp, rse, overblur = read_results_table()

    input_path = ""
    reference_path = ""
    output_path = ""
    convolved_output_path = ""
    error_map_path = ""

    filename_prefix = "group_%03d_channel_%d" % (group_index, channel_index)
    if CONFIG["save_image_outputs"]:
        input_path = save_image(input_imp, group_index, "group_%03d_input.tif" % group_index)
        reference_path = save_image(
            ref_imp, group_index, filename_prefix + "_reference.tif"
        )
        output_path = save_image(sr_imp, group_index, filename_prefix + "_output.tif")

        convolved_title = sr_title + " - Convolved with RSF"
        convolved_imp = WindowManager.getImage(convolved_title)
        if convolved_imp is None:
            raise RuntimeError("Could not find SQUIRREL convolved output image: " + convolved_title)
        convolved_output_path = save_image(
            convolved_imp, group_index, filename_prefix + "_output_convolved.tif"
        )

    if CONFIG["save_error_maps"]:
        err_title = sr_title + " - Resolution Scaled Error-Map"
        err_imp = WindowManager.getImage(err_title)
        if err_imp is None:
            raise RuntimeError("Could not find SQUIRREL error-map image: " + err_title)
        error_map_path = save_image(
            err_imp, group_index, filename_prefix + "_rse_error_map.tif"
        )

    close_all_images()
    return {
        "group": group_index,
        "channel": channel_index,
        "input_slice": input_slice,
        "sr_slice": sr_slice,
        "ref_slice": ref_slice,
        "rsp": rsp,
        "rse": rse,
        "overblur": overblur,
        "input_path": input_path,
        "reference_path": reference_path,
        "output_path": output_path,
        "convolved_output_path": convolved_output_path,
        "error_map_path": error_map_path,
    }


def main():
    out_dir = CONFIG["output_dir"]
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    summary_csv = CONFIG["summary_csv"]
    log_csv = CONFIG["progress_csv"]
    header = [
        "group", "channel", "input_slice", "sr_slice", "ref_slice",
        "RSP", "RSE", "overblur", "input_path", "reference_path",
        "output_path", "convolved_output_path", "rse_error_map_path"
    ]
    write_text(summary_csv, ",".join(header) + "\n")
    write_text(log_csv, ",".join(header) + "\n")

    source = IJ.openImage(CONFIG["input_stack"])
    if source is None:
        raise RuntimeError("Could not open stack: " + CONFIG["input_stack"])
    n_slices = source.getStackSize()
    if n_slices % 5 != 0:
        raise RuntimeError("Stack slice count must be a multiple of 5; got %d" % n_slices)

    groups = n_slices // 5
    if CONFIG["max_groups"] > 0:
        groups = min(groups, CONFIG["max_groups"])

    for group_index in range(1, groups + 1):
        for channel_index in (1, 2):
            print("Running SQUIRREL group %d/%d channel %d" % (group_index, groups, channel_index))
            result = run_pair(source, group_index, channel_index)
            row = [
                result["group"],
                result["channel"],
                result["input_slice"],
                result["sr_slice"],
                result["ref_slice"],
                "%.12g" % result["rsp"],
                "%.12g" % result["rse"],
                result["overblur"],
                result["input_path"],
                result["reference_path"],
                result["output_path"],
                result["convolved_output_path"],
                result["error_map_path"],
            ]
            line = ",".join(csv_escape(x) for x in row) + "\n"
            append_text(summary_csv, line)
            append_text(log_csv, line)

    write_text(CONFIG["done_file"], "done\n")


try:
    main()
    System.exit(0)
except Exception:
    msg = traceback.format_exc()
    print(msg)
    write_text(CONFIG["fail_file"], msg)
    System.exit(1)
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NanoJ-SQUIRREL (No GPU) RSE/RSP on a 5-slice-group stack."
    )
    parser.add_argument(
        "input_stack",
        nargs="?",
        default=str(DEFAULT_STACK),
        help="Input TIFF stack. Defaults to the test stack from the prompt.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory. Default: <input_stem>_squirrel_results next to the stack.",
    )
    parser.add_argument(
        "--imagej-dir",
        default=str(DEFAULT_IMAGEJ_DIR),
        help="Path to the Fiji/ImageJ directory containing ImageJ-win64.exe.",
    )
    parser.add_argument(
        "--heap",
        default="8g",
        help="Java heap size, e.g. 8g, 16g, 32000m. Default: 8g.",
    )
    parser.add_argument(
        "--pixel-size-nm",
        type=float,
        default=20.0,
        help=(
            "Pixel size in nanometers written into ImageJ calibration before "
            "running SQUIRREL. Use 0 to keep the TIFF's original calibration. Default: 20."
        ),
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="For testing, limit the number of 5-slice groups. 0 means all groups.",
    )
    parser.add_argument(
        "--batch-root",
        default=None,
        help=(
            "Run batch mode over stacks named <resolution>_<poisson_noise>.tif "
            "inside this folder."
        ),
    )
    parser.add_argument(
        "--resolutions",
        default="225,245,265",
        help="Comma-separated resolution values for batch mode. Default: 225,245,265.",
    )
    parser.add_argument(
        "--poisson-noises",
        default="0.040,0.120,0.200,0.280",
        help="Comma-separated Poisson noise values for batch mode.",
    )
    parser.add_argument(
        "--batch-output-dir",
        default=None,
        help="Batch output directory. Default: <batch-root>/squirrel_batch_results.",
    )
    parser.add_argument(
        "--combined-excel",
        default=None,
        help="Combined Excel path for batch mode. Default: <batch-output-dir>/squirrel_batch_summary.xlsx.",
    )
    parser.add_argument(
        "--save-error-maps",
        action="store_true",
        help="Explicitly save per-pair RSE error-map TIFFs. This is already the default.",
    )
    parser.add_argument(
        "--no-save-error-maps",
        action="store_true",
        help="Do not save per-pair RSE error-map TIFFs.",
    )
    parser.add_argument(
        "--save-image-outputs",
        action="store_true",
        help=(
            "Explicitly save Input, Reference, Output, and RSF-convolved Output TIFFs. "
            "This is already the default."
        ),
    )
    parser.add_argument(
        "--no-save-image-outputs",
        action="store_true",
        help="Do not save Input, Reference, Output, or RSF-convolved Output TIFFs.",
    )
    parser.add_argument(
        "--do-registration",
        action="store_true",
        help=(
            "Enable SQUIRREL registration. Disabled by default because this ImageJ "
            "install is missing NanoJ-Core classes needed by the registration path."
        ),
    )
    parser.add_argument(
        "--no-border-control",
        action="store_true",
        help="Disable SQUIRREL black-border crop control.",
    )
    parser.add_argument(
        "--no-smart-boundary",
        action="store_true",
        help="Disable SQUIRREL smart boundary for RSF sigma.",
    )
    parser.add_argument(
        "--frame-purge",
        action="store_true",
        help="Enable SQUIRREL empty-frame purge.",
    )
    parser.add_argument(
        "--max-expected-misalignment",
        type=int,
        default=0,
        help="Registration max expected misalignment. Only used with --do-registration.",
    )
    parser.add_argument(
        "--show-positive-negative",
        action="store_true",
        help="Ask SQUIRREL to keep positive/negative error-map contributions.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0,
        help="Overall Fiji timeout in seconds. 0 means no timeout.",
    )
    parser.add_argument(
        "--keep-inner-script",
        action="store_true",
        help="Keep the generated Jython runner script for debugging.",
    )
    parser.add_argument(
        "--no-clean-stubs",
        action="store_true",
        help="Do not remove stale ImageJ single-instance .stub files from TEMP before launch.",
    )
    return parser.parse_args()


def find_one(pattern: str, root: Path) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern!r} under {root}")
    return matches[0]


def find_java(imagej_dir: Path) -> Path:
    candidates = sorted((imagej_dir / "java").glob(r"win64/*/bin/java.exe"))
    if not candidates:
        raise FileNotFoundError(f"Could not find bundled java.exe under {imagej_dir / 'java'}")
    return candidates[0]


def build_java_command(imagej_dir: Path, heap: str, runner_script: Path) -> list[str]:
    java = find_java(imagej_dir)
    launcher_jar = find_one("imagej-launcher-*.jar", imagej_dir / "jars")
    imagej_exe = imagej_dir / "ImageJ-win64.exe"
    java_library_path = os.pathsep.join(
        [str(imagej_dir / "lib" / "win64"), str(imagej_dir / "mm" / "win64")]
    )

    return [
        str(java),
        "-Dpython.cachedir.skip=true",
        f"-Dplugins.dir={imagej_dir}",
        f"-Xmx{heap}",
        "-Dapple.awt.UIElement=true",
        "-classpath",
        str(launcher_jar),
        f"-Dimagej.dir={imagej_dir}",
        f"-Dij.dir={imagej_dir}",
        f"-Dfiji.dir={imagej_dir}",
        "-Dfiji.defaultLibPath=bin/server/jvm.dll",
        f"-Dfiji.executable={imagej_exe}",
        f"-Dij.executable={imagej_exe}",
        f"-Djava.library.path={java_library_path}",
        "-Dscijava.context.strict=false",
        "net.imagej.launcher.ClassLauncher",
        "-ijjarpath",
        "jars",
        "-ijjarpath",
        "plugins",
        "net.imagej.Main",
        "--run",
        str(runner_script),
    ]


def write_runner(config: dict, path: Path) -> None:
    config_json = json.dumps(config, ensure_ascii=False)
    script = JYTHON_RUNNER.replace("__CONFIG_JSON__", config_json)
    path.write_text(script, encoding="utf-8")


def clean_imagej_stubs() -> None:
    temp = os.environ.get("TEMP") or os.environ.get("TMP")
    if not temp:
        return
    for stub in Path(temp).glob("ImageJ-*.stub"):
        try:
            stub.chmod(stat.S_IWRITE)
            stub.unlink()
        except OSError:
            pass


def run_fiji(
    command: list[str],
    cwd: Path,
    log_path: Path,
    done_file: Path,
    fail_file: Path,
    timeout: float,
) -> int:
    start = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

        while True:
            rc = proc.poll()
            if rc is not None:
                return rc

            if done_file.exists():
                try:
                    return proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
                    return 0

            if fail_file.exists():
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                return 1

            if timeout and (time.time() - start) > timeout:
                proc.kill()
                raise TimeoutError(f"Fiji timed out after {timeout} seconds")

            time.sleep(1)


def run_one_stack(args: argparse.Namespace) -> Path:
    input_stack = Path(args.input_stack).expanduser().resolve()
    imagej_dir = Path(args.imagej_dir).expanduser().resolve()
    if not input_stack.exists():
        raise FileNotFoundError(input_stack)
    if not imagej_dir.exists():
        raise FileNotFoundError(imagej_dir)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_stack.with_name(input_stack.stem + "_squirrel_results").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runner_script = output_dir / "_run_nanoj_squirrel_inner.py"
    done_file = output_dir / "_done.txt"
    fail_file = output_dir / "_failed.txt"
    log_path = output_dir / "fiji_run.log"
    summary_csv = output_dir / "squirrel_rse_rsp_summary.csv"
    progress_csv = output_dir / "_progress_rows.csv"

    for stale in (done_file, fail_file):
        if stale.exists():
            stale.unlink()

    config = {
        "input_stack": str(input_stack),
        "output_dir": str(output_dir),
        "summary_csv": str(summary_csv),
        "progress_csv": str(progress_csv),
        "done_file": str(done_file),
        "fail_file": str(fail_file),
        "max_groups": int(args.max_groups),
        "pixel_size_nm": float(args.pixel_size_nm),
        "save_error_maps": args.save_error_maps or not args.no_save_error_maps,
        "save_image_outputs": args.save_image_outputs or not args.no_save_image_outputs,
        "do_registration": bool(args.do_registration),
        "border_control": not args.no_border_control,
        "smart_boundary": not args.no_smart_boundary,
        "frame_purge": bool(args.frame_purge),
        "max_expected_misalignment": int(args.max_expected_misalignment),
        "show_positive_negative": bool(args.show_positive_negative),
    }
    write_runner(config, runner_script)

    command = build_java_command(imagej_dir, args.heap, runner_script)
    if not args.no_clean_stubs:
        clean_imagej_stubs()
    print(f"Running Fiji/ImageJ from: {imagej_dir}")
    print(f"Input stack: {input_stack}")
    print(f"Output directory: {output_dir}")
    if args.do_registration:
        print("Warning: --do-registration requires NanoJ-Core classes; this local install failed that path in testing.")

    rc = run_fiji(command, imagej_dir, log_path, done_file, fail_file, args.timeout)

    if fail_file.exists():
        message = fail_file.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"Fiji runner failed:\n{message}\nSee log: {log_path}")
    if not done_file.exists():
        raise RuntimeError(f"Fiji exited with code {rc} before writing done file. See log: {log_path}")
    if not summary_csv.exists():
        raise RuntimeError(f"Done file exists but summary CSV was not written. See log: {log_path}")

    if not args.keep_inner_script:
        try:
            runner_script.unlink()
        except OSError:
            pass

    print(f"Done. Summary CSV: {summary_csv}")
    if not args.no_save_image_outputs or not args.no_save_error_maps:
        print(f"Saved images: {output_dir / 'saved_images'}")
    print(f"Fiji log: {log_path}")
    return summary_csv


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_combined_excel(rows: list[dict[str, str]], excel_path: Path) -> None:
    from openpyxl import Workbook

    excel_path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "RSE_RSP"

    preferred = [
        "resolution",
        "poisson_noise",
        "dataset",
        "source_stack",
        "group",
        "channel",
        "input_slice",
        "sr_slice",
        "ref_slice",
        "RSP",
        "RSE",
        "overblur",
        "input_path",
        "reference_path",
        "output_path",
        "convolved_output_path",
        "rse_error_map_path",
    ]
    extra = sorted({key for row in rows for key in row.keys()} - set(preferred))
    headers = preferred + extra
    ws.append(headers)

    for row in rows:
        ws.append([row.get(header, "") for header in headers])

    for column_cells in ws.columns:
        max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
        ws.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 10), 80)

    wb.save(str(excel_path))


def run_batch(args: argparse.Namespace) -> Path:
    batch_root = Path(args.batch_root).expanduser().resolve()
    if not batch_root.exists():
        raise FileNotFoundError(batch_root)

    batch_output_dir = (
        Path(args.batch_output_dir).expanduser().resolve()
        if args.batch_output_dir
        else batch_root / "squirrel_batch_results"
    )
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    excel_path = (
        Path(args.combined_excel).expanduser().resolve()
        if args.combined_excel
        else batch_output_dir / "squirrel_batch_summary.xlsx"
    )

    all_rows: list[dict[str, str]] = []
    resolutions = parse_csv_list(args.resolutions)
    poisson_noises = parse_csv_list(args.poisson_noises)

    for resolution_selected in resolutions:
        for poisson_noise_selected in poisson_noises:
            dataset = f"{resolution_selected}_{poisson_noise_selected}"
            stack_path = batch_root / f"{dataset}.tif"
            if not stack_path.exists():
                print(f"Skipping missing stack: {stack_path}")
                continue

            dataset_output_dir = batch_output_dir / dataset
            print(f"\n=== Running dataset {dataset} ===")

            one_args = argparse.Namespace(**vars(args))
            one_args.input_stack = str(stack_path)
            one_args.output_dir = str(dataset_output_dir)
            summary_csv = run_one_stack(one_args)

            with summary_csv.open("r", newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    row = dict(row)
                    row["resolution"] = resolution_selected
                    row["poisson_noise"] = poisson_noise_selected
                    row["dataset"] = dataset
                    row["source_stack"] = str(stack_path)
                    all_rows.append(row)

            write_combined_excel(all_rows, excel_path)

    if not all_rows:
        raise RuntimeError(f"No batch rows were generated from {batch_root}")

    write_combined_excel(all_rows, excel_path)
    print(f"\nBatch done. Combined Excel: {excel_path}")
    return excel_path


def main() -> int:
    args = parse_args()
    if args.batch_root:
        run_batch(args)
    else:
        run_one_stack(args)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())

    sys.argv = [
        "run_nanoj_squirrel_stack.py",

        "--batch-root",
        str(DESM_ROOT / "visualization" / "Experimental_data_eval" / "Micro_Mito" / "resized results"),

        "--batch-output-dir",
        str(DESM_ROOT / "visualization" / "Experimental_data_eval" / "Micro_Mito" / "resized results" / "squirrel_batch_results"),

        "--combined-excel",
        str(
            DESM_ROOT
            / "visualization"
            / "Experimental_data_eval"
            / "Micro_Mito"
            / "resized results"
            / "squirrel_batch_results"
            / "squirrel_batch_summary.xlsx"
        ),

        "--resolutions",
        "225,245,265",

        "--poisson-noises",
        "0.040,0.120,0.200,0.280",

        str(DEFAULT_STACK),

        "-o",
        str(DESM_ROOT / "visualization" / "Experimental_data_eval" / "Micro_Mito" / "test" / "squirrel_results"),

        "--imagej-dir",
        str(DEFAULT_IMAGEJ_DIR),

        # 测试时只跑前 1 组；正式跑全部时删掉这两行
        #"--max-groups",
        #"1",

        "--heap",
        "8g",

        "--pixel-size-nm",
        "20",

        "--save-image-outputs",
        "--save-error-maps",
    ]

    main()
