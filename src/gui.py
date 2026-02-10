"""
GUI for EZ-FBX2VRM using CustomTkinter.

Provides a simple interface for selecting an FBX file, entering VRM metadata,
and converting to VRM format.
"""

import logging
import threading
import traceback
from pathlib import Path

try:
    import customtkinter as ctk
except ImportError:
    ctk = None

logger = logging.getLogger(__name__)


LICENSE_OPTIONS = [
    "CC0",
    "CC_BY",
    "CC_BY_NC",
    "CC_BY_SA",
    "CC_BY_NC_SA",
    "CC_BY_ND",
    "CC_BY_NC_ND",
    "Redistribution_Prohibited",
    "Other",
]

PERMISSION_OPTIONS = ["Allow", "Disallow"]
USER_OPTIONS = ["OnlyAuthor", "ExplicitlyLicensedPerson", "Everyone"]


class App:
    """Main application window."""

    def __init__(self):
        if ctk is None:
            raise ImportError(
                "customtkinter is not installed. Install it with: pip install customtkinter"
            )

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("EZ-FBX2VRM - FBX to VRM Converter")
        self.root.geometry("720x780")
        self.root.minsize(600, 700)

        self._input_path = ""
        self._output_path = ""
        self._converting = False

        self._build_ui()

    def _build_ui(self):
        # Main scrollable frame
        main = ctk.CTkFrame(self.root)
        main.pack(fill="both", expand=True, padx=16, pady=16)

        # --- Title ---
        title = ctk.CTkLabel(
            main, text="EZ-FBX2VRM",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        title.pack(pady=(8, 0))

        subtitle = ctk.CTkLabel(
            main, text="Convert Mixamo-rigged FBX to VRM (no Blender/Unity needed)",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        )
        subtitle.pack(pady=(0, 12))

        # --- File Selection ---
        file_frame = ctk.CTkFrame(main)
        file_frame.pack(fill="x", padx=8, pady=4)

        ctk.CTkLabel(file_frame, text="Input FBX File:", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(8, 2)
        )

        input_row = ctk.CTkFrame(file_frame, fg_color="transparent")
        input_row.pack(fill="x", padx=8, pady=2)

        self._input_entry = ctk.CTkEntry(input_row, placeholder_text="Select FBX file...")
        self._input_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))

        ctk.CTkButton(input_row, text="Browse", width=80, command=self._browse_input).pack(
            side="right"
        )

        ctk.CTkLabel(file_frame, text="Output VRM File:", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(8, 2)
        )

        output_row = ctk.CTkFrame(file_frame, fg_color="transparent")
        output_row.pack(fill="x", padx=8, pady=(2, 8))

        self._output_entry = ctk.CTkEntry(output_row, placeholder_text="Output path (auto-generated if empty)")
        self._output_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))

        ctk.CTkButton(output_row, text="Browse", width=80, command=self._browse_output).pack(
            side="right"
        )

        # --- VRM Metadata ---
        meta_frame = ctk.CTkFrame(main)
        meta_frame.pack(fill="x", padx=8, pady=4)

        ctk.CTkLabel(meta_frame, text="VRM Metadata", font=ctk.CTkFont(size=15, weight="bold")).pack(
            anchor="w", padx=8, pady=(8, 4)
        )

        # Title
        self._meta_title = self._add_field(meta_frame, "Title:", "My Model")
        # Author
        self._meta_author = self._add_field(meta_frame, "Author:", "")
        # Version
        self._meta_version = self._add_field(meta_frame, "Version:", "1.0")

        # License
        license_row = ctk.CTkFrame(meta_frame, fg_color="transparent")
        license_row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(license_row, text="License:", width=120, anchor="w").pack(side="left")
        self._meta_license = ctk.CTkComboBox(license_row, values=LICENSE_OPTIONS, width=200)
        self._meta_license.set("CC0")
        self._meta_license.pack(side="left", padx=4)

        # Allowed user
        user_row = ctk.CTkFrame(meta_frame, fg_color="transparent")
        user_row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(user_row, text="Allowed User:", width=120, anchor="w").pack(side="left")
        self._meta_allowed_user = ctk.CTkComboBox(user_row, values=USER_OPTIONS, width=200)
        self._meta_allowed_user.set("Everyone")
        self._meta_allowed_user.pack(side="left", padx=4)

        # Violent usage
        v_row = ctk.CTkFrame(meta_frame, fg_color="transparent")
        v_row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(v_row, text="Violent Usage:", width=120, anchor="w").pack(side="left")
        self._meta_violent = ctk.CTkComboBox(v_row, values=PERMISSION_OPTIONS, width=200)
        self._meta_violent.set("Disallow")
        self._meta_violent.pack(side="left", padx=4)

        # Sexual usage
        s_row = ctk.CTkFrame(meta_frame, fg_color="transparent")
        s_row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(s_row, text="Sexual Usage:", width=120, anchor="w").pack(side="left")
        self._meta_sexual = ctk.CTkComboBox(s_row, values=PERMISSION_OPTIONS, width=200)
        self._meta_sexual.set("Disallow")
        self._meta_sexual.pack(side="left", padx=4)

        # Commercial usage
        c_row = ctk.CTkFrame(meta_frame, fg_color="transparent")
        c_row.pack(fill="x", padx=8, pady=(2, 8))
        ctk.CTkLabel(c_row, text="Commercial Usage:", width=120, anchor="w").pack(side="left")
        self._meta_commercial = ctk.CTkComboBox(c_row, values=PERMISSION_OPTIONS, width=200)
        self._meta_commercial.set("Disallow")
        self._meta_commercial.pack(side="left", padx=4)

        # --- Convert Button ---
        self._convert_btn = ctk.CTkButton(
            main, text="Convert to VRM", height=42,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self._start_conversion,
        )
        self._convert_btn.pack(fill="x", padx=8, pady=8)

        # --- Progress ---
        self._progress_bar = ctk.CTkProgressBar(main)
        self._progress_bar.pack(fill="x", padx=8, pady=4)
        self._progress_bar.set(0)

        self._status_label = ctk.CTkLabel(
            main, text="Ready", text_color="gray",
            font=ctk.CTkFont(size=12),
        )
        self._status_label.pack(anchor="w", padx=8)

        # --- Log ---
        self._log_text = ctk.CTkTextbox(main, height=140, font=ctk.CTkFont(size=11))
        self._log_text.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self._log_text.configure(state="disabled")

    def _add_field(self, parent, label: str, default: str) -> ctk.CTkEntry:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row, text=label, width=120, anchor="w").pack(side="left")
        entry = ctk.CTkEntry(row, placeholder_text=default)
        entry.pack(side="left", fill="x", expand=True, padx=4)
        return entry

    def _browse_input(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select FBX File",
            filetypes=[("FBX files", "*.fbx"), ("All files", "*.*")],
        )
        if path:
            self._input_entry.delete(0, "end")
            self._input_entry.insert(0, path)
            # Auto-generate output path
            if not self._output_entry.get():
                out = str(Path(path).with_suffix('.vrm'))
                self._output_entry.delete(0, "end")
                self._output_entry.insert(0, out)

    def _browse_output(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            title="Save VRM File",
            defaultextension=".vrm",
            filetypes=[("VRM files", "*.vrm"), ("All files", "*.*")],
        )
        if path:
            self._output_entry.delete(0, "end")
            self._output_entry.insert(0, path)

    def _log(self, message: str):
        """Append a message to the log area (thread-safe)."""
        def _do():
            self._log_text.configure(state="normal")
            self._log_text.insert("end", message + "\n")
            self._log_text.see("end")
            self._log_text.configure(state="disabled")
        self.root.after(0, _do)

    def _set_progress(self, value: float, message: str = ""):
        def _do():
            self._progress_bar.set(value)
            if message:
                self._status_label.configure(text=message)
        self.root.after(0, _do)

    def _get_meta(self) -> dict:
        return {
            "title": self._meta_title.get() or "My Model",
            "author": self._meta_author.get() or "Unknown",
            "version": self._meta_version.get() or "1.0",
            "license": self._meta_license.get(),
            "allowedUser": self._meta_allowed_user.get(),
            "violentUsage": self._meta_violent.get(),
            "sexualUsage": self._meta_sexual.get(),
            "commercialUsage": self._meta_commercial.get(),
        }

    def _start_conversion(self):
        if self._converting:
            return

        input_path = self._input_entry.get().strip()
        if not input_path:
            self._log("ERROR: No input file selected.")
            return

        output_path = self._output_entry.get().strip()
        if not output_path:
            output_path = str(Path(input_path).with_suffix('.vrm'))
            self._output_entry.delete(0, "end")
            self._output_entry.insert(0, output_path)

        self._converting = True
        self._convert_btn.configure(state="disabled", text="Converting...")
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")
        self._set_progress(0, "Starting...")

        meta = self._get_meta()

        def _run():
            try:
                from .converter import convert_fbx_to_vrm

                def _callback(msg, pct):
                    self._log(msg)
                    self._set_progress(pct, msg)

                convert_fbx_to_vrm(input_path, output_path, meta=meta, callback=_callback)
                self._log(f"\nConversion successful!\nOutput: {output_path}")
                self._set_progress(1.0, "Conversion complete!")
            except Exception as e:
                self._log(f"\nERROR: {e}")
                self._log(traceback.format_exc())
                self._set_progress(0, f"Error: {e}")
            finally:
                def _done():
                    self._converting = False
                    self._convert_btn.configure(state="normal", text="Convert to VRM")
                self.root.after(0, _done)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def run(self):
        """Start the application main loop."""
        self.root.mainloop()


def main():
    """Entry point for the GUI application."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app = App()
    app.run()


if __name__ == "__main__":
    main()
