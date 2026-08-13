import subprocess


_real_check_output = subprocess.check_output


def _check_output_without_macos_font_scan(cmd, *args, **kwargs):
    if (
        isinstance(cmd, (list, tuple))
        and len(cmd) >= 3
        and cmd[0] == "system_profiler"
        and "SPFontsDataType" in cmd
    ):
        return (
            b"<?xml version='1.0'?>"
            b"<plist version='1.0'><array><dict>"
            b"<key>_items</key><array/>"
            b"</dict></array></plist>"
        )
    return _real_check_output(cmd, *args, **kwargs)


subprocess.check_output = _check_output_without_macos_font_scan
