import sys
import json
import subprocess
import os

EXIFTOOL_PATH = os.environ.get("EXIFTOOL") or os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "exiftool-13.59_64", "exiftool.exe")
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def read_metadata_folder(folder):
    """
    Read the geolocation EXIF / DJI maker-note fields for every image in a
    folder in a single exiftool call. Returns a list of dicts, each carrying a
    "SourceFile" plus the requested tags (numeric, thanks to -n).
    """
    result = subprocess.run(
        [
            EXIFTOOL_PATH,
            "-j",   # JSON output
            "-n",   # numeric values (decimal GPS, degrees, etc.)
            "-GPSLatitude",
            "-GPSLongitude",
            "-GPSAltitude",
            "-FocalLength",
            "-GimbalYawDegree",
            "-FlightYawDegree",
            folder,
        ],
        capture_output=True,
        text=True,
    )

    # exiftool returns a non-zero code when *some* files can't be read, but
    # still emits JSON for the ones it could — only bail if we got nothing.
    if not result.stdout.strip():
        raise RuntimeError(
            f"exiftool produced no output: {result.stderr.strip()}"
        )

    return json.loads(result.stdout)


def main():
    if len(sys.argv) < 2:
        raise RuntimeError("usage: validate_metadata.py <folder> [yaw_source]")

    folder = sys.argv[1]
    yaw_source = sys.argv[2] if len(sys.argv) > 2 else "gimbal"

    required = ["GPSLatitude", "GPSLongitude", "FocalLength", "GPSAltitude"]
    required.append(
        "GimbalYawDegree" if yaw_source == "gimbal" else "FlightYawDegree"
    )

    metadata = read_metadata_folder(folder)

    issues = []
    checked = 0

    for entry in metadata:
        source = entry.get("SourceFile", "")
        name = os.path.basename(source)

        # skip anything that isn't an image we accept
        if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
            continue

        checked += 1

        # a tag is "missing" when exiftool didn't emit the key at all
        missing = [tag for tag in required if entry.get(tag) is None]
        if missing:
            issues.append({"file": name, "missing": missing})

    print(json.dumps({
        "ok": len(issues) == 0,
        "checked": checked,
        "required": required,
        "issues": issues,
    }))


if __name__ == "__main__":
    main()
