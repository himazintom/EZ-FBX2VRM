"""
GUI for EZ-FBX2VRM using CustomTkinter.

Three-tab interface:
  1. Convert  - FBX to VRM conversion
  2. Preview  - 3D model viewer with orbit camera
  3. MoCap    - Webcam motion capture synced to 3D model
"""

import logging
import threading
import time
import traceback
from pathlib import Path

import numpy as np

try:
    import customtkinter as ctk
except ImportError:
    ctk = None

try:
    from PIL import Image as PILImage, ImageTk
except ImportError:
    PILImage = None

logger = logging.getLogger(__name__)

LICENSE_OPTIONS = [
    "CC0", "CC_BY", "CC_BY_NC", "CC_BY_SA", "CC_BY_NC_SA",
    "CC_BY_ND", "CC_BY_NC_ND", "Redistribution_Prohibited", "Other",
]
PERMISSION_OPTIONS = ["Allow", "Disallow"]
USER_OPTIONS = ["OnlyAuthor", "ExplicitlyLicensedPerson", "Everyone"]

# Target FPS for render/preview loops
TARGET_FPS = 30
FRAME_MS = int(1000 / TARGET_FPS)


class App:
    """Main application window with tabbed interface."""

    def __init__(self):
        if ctk is None:
            raise ImportError("customtkinter is required: pip install customtkinter")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("EZ-FBX2VRM")
        self.root.geometry("1000x750")
        self.root.minsize(800, 650)

        self._converting = False
        self._loading_model = False  # Guard against concurrent model loading

        # Shared model data
        self._fbx_data = None
        self._renderer = None
        self._camera = None
        self._mocap = None
        self._pose_solver = None
        self._bone_mapping = None

        # Preview state
        self._preview_running = False
        self._preview_image_label = None
        self._preview_after_id = None

        # MoCap state
        self._mocap_running = False
        self._mocap_after_id = None

        # Mouse state
        self._mouse_last_x = 0
        self._mouse_last_y = 0

        self._build_ui()

    # ──────────────────── UI Construction ────────────────────

    def _build_ui(self):
        # Title bar
        header = ctk.CTkFrame(self.root, height=50, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(8, 0))
        ctk.CTkLabel(
            header, text="EZ-FBX2VRM",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(side="left", padx=8)
        ctk.CTkLabel(
            header, text="FBX to VRM Converter + Motion Capture",
            font=ctk.CTkFont(size=12), text_color="gray",
        ).pack(side="left", padx=8)

        # Tabs
        self._tabview = ctk.CTkTabview(self.root, anchor="nw")
        self._tabview.pack(fill="both", expand=True, padx=12, pady=8)

        self._tab_convert = self._tabview.add("Convert")
        self._tab_preview = self._tabview.add("Preview")
        self._tab_mocap = self._tabview.add("MoCap")

        self._build_convert_tab()
        self._build_preview_tab()
        self._build_mocap_tab()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──── Convert Tab ────

    def _build_convert_tab(self):
        tab = self._tab_convert

        # File selection
        file_frame = ctk.CTkFrame(tab)
        file_frame.pack(fill="x", padx=8, pady=4)

        ctk.CTkLabel(file_frame, text="Input FBX:", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(8, 2))
        input_row = ctk.CTkFrame(file_frame, fg_color="transparent")
        input_row.pack(fill="x", padx=8, pady=2)
        self._input_entry = ctk.CTkEntry(input_row, placeholder_text="Select FBX file...")
        self._input_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkButton(input_row, text="Browse", width=80, command=self._browse_input).pack(side="right")

        ctk.CTkLabel(file_frame, text="Output VRM:", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(8, 2))
        output_row = ctk.CTkFrame(file_frame, fg_color="transparent")
        output_row.pack(fill="x", padx=8, pady=(2, 8))
        self._output_entry = ctk.CTkEntry(output_row, placeholder_text="Output path (auto)")
        self._output_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkButton(output_row, text="Browse", width=80, command=self._browse_output).pack(side="right")

        # Metadata
        meta_frame = ctk.CTkFrame(tab)
        meta_frame.pack(fill="x", padx=8, pady=4)
        ctk.CTkLabel(meta_frame, text="VRM Metadata", font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=8, pady=(8, 4))

        self._meta_title = self._add_field(meta_frame, "Title:", "My Model")
        self._meta_author = self._add_field(meta_frame, "Author:", "")
        self._meta_version = self._add_field(meta_frame, "Version:", "1.0")

        self._meta_license = self._add_combo(meta_frame, "License:", LICENSE_OPTIONS, "CC0")
        self._meta_allowed_user = self._add_combo(meta_frame, "Allowed User:", USER_OPTIONS, "Everyone")
        self._meta_violent = self._add_combo(meta_frame, "Violent Usage:", PERMISSION_OPTIONS, "Disallow")
        self._meta_sexual = self._add_combo(meta_frame, "Sexual Usage:", PERMISSION_OPTIONS, "Disallow")
        self._meta_commercial = self._add_combo(meta_frame, "Commercial:", PERMISSION_OPTIONS, "Disallow")

        # Convert button
        self._convert_btn = ctk.CTkButton(
            tab, text="Convert to VRM", height=40,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._start_conversion,
        )
        self._convert_btn.pack(fill="x", padx=8, pady=8)

        # Progress
        self._progress_bar = ctk.CTkProgressBar(tab)
        self._progress_bar.pack(fill="x", padx=8, pady=4)
        self._progress_bar.set(0)
        self._status_label = ctk.CTkLabel(tab, text="Ready", text_color="gray", font=ctk.CTkFont(size=11))
        self._status_label.pack(anchor="w", padx=8)

        # Log
        self._log_text = ctk.CTkTextbox(tab, height=120, font=ctk.CTkFont(size=11))
        self._log_text.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self._log_text.configure(state="disabled")

    # ──── Preview Tab ────

    def _build_preview_tab(self):
        tab = self._tab_preview

        # Toolbar
        toolbar = ctk.CTkFrame(tab, height=40, fg_color="transparent")
        toolbar.pack(fill="x", padx=4, pady=4)

        ctk.CTkButton(toolbar, text="Load FBX", width=100, command=self._preview_load_model).pack(
            side="left", padx=4)
        ctk.CTkButton(toolbar, text="Reset View", width=100, command=self._preview_reset_view).pack(
            side="left", padx=4)

        self._preview_info = ctk.CTkLabel(toolbar, text="No model loaded", text_color="gray",
                                          font=ctk.CTkFont(size=11))
        self._preview_info.pack(side="right", padx=8)

        # Help text
        ctk.CTkLabel(tab, font=ctk.CTkFont(size=11), text_color="#888888",
                     text="Left drag: Rotate | Right drag: Zoom | Shift+Left drag: Pan | Scroll: Zoom"
                     ).pack(anchor="w", padx=8)

        # Viewport
        self._preview_canvas = ctk.CTkLabel(tab, text="")
        self._preview_canvas.pack(fill="both", expand=True, padx=4, pady=4)

        # Bind mouse events to the viewport label's internal tkinter widget
        widget = self._preview_canvas
        widget.bind("<ButtonPress-1>", self._on_preview_press)
        widget.bind("<ButtonPress-3>", self._on_preview_press)
        widget.bind("<B1-Motion>", self._on_preview_drag)
        widget.bind("<B3-Motion>", self._on_preview_drag)
        widget.bind("<ButtonRelease-1>", self._on_preview_release)
        widget.bind("<ButtonRelease-3>", self._on_preview_release)
        widget.bind("<MouseWheel>", self._on_preview_scroll)
        # Linux scroll
        widget.bind("<Button-4>", lambda e: self._on_preview_scroll_linux(e, 1))
        widget.bind("<Button-5>", lambda e: self._on_preview_scroll_linux(e, -1))

        self._preview_drag_button = None
        self._preview_shift = False

    # ──── MoCap Tab ────

    def _build_mocap_tab(self):
        tab = self._tab_mocap

        # Controls
        ctrl = ctk.CTkFrame(tab, height=45, fg_color="transparent")
        ctrl.pack(fill="x", padx=4, pady=4)

        ctk.CTkButton(ctrl, text="Load FBX", width=100, command=self._mocap_load_model).pack(
            side="left", padx=4)

        self._mocap_start_btn = ctk.CTkButton(
            ctrl, text="Start Camera", width=120,
            fg_color="#28a745", hover_color="#218838",
            command=self._mocap_toggle,
        )
        self._mocap_start_btn.pack(side="left", padx=4)

        ctk.CTkLabel(ctrl, text="Camera:", font=ctk.CTkFont(size=11)).pack(side="left", padx=(12, 2))
        self._cam_index_entry = ctk.CTkEntry(ctrl, width=40, placeholder_text="0")
        self._cam_index_entry.insert(0, "0")
        self._cam_index_entry.pack(side="left", padx=2)

        self._mirror_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(ctrl, text="Mirror", variable=self._mirror_var, width=60,
                        command=self._mocap_update_mirror).pack(side="left", padx=8)

        self._mocap_fps_label = ctk.CTkLabel(ctrl, text="FPS: --", text_color="gray",
                                             font=ctk.CTkFont(size=11))
        self._mocap_fps_label.pack(side="right", padx=8)

        self._mocap_status = ctk.CTkLabel(ctrl, text="Camera off", text_color="gray",
                                          font=ctk.CTkFont(size=11))
        self._mocap_status.pack(side="right", padx=8)

        # Split view: camera on left, 3D model on right
        split = ctk.CTkFrame(tab, fg_color="transparent")
        split.pack(fill="both", expand=True, padx=4, pady=4)
        split.grid_columnconfigure(0, weight=1)
        split.grid_columnconfigure(1, weight=1)
        split.grid_rowconfigure(0, weight=1)

        # Camera view
        cam_frame = ctk.CTkFrame(split)
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        ctk.CTkLabel(cam_frame, text="Camera", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#aaa").pack(anchor="w", padx=4, pady=2)
        self._mocap_cam_label = ctk.CTkLabel(cam_frame, text="")
        self._mocap_cam_label.pack(fill="both", expand=True, padx=2, pady=2)

        # Model view
        model_frame = ctk.CTkFrame(split)
        model_frame.grid(row=0, column=1, sticky="nsew", padx=(2, 0))
        ctk.CTkLabel(model_frame, text="Model", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#aaa").pack(anchor="w", padx=4, pady=2)
        self._mocap_model_label = ctk.CTkLabel(model_frame, text="")
        self._mocap_model_label.pack(fill="both", expand=True, padx=2, pady=2)

    # ──────────────────── Helpers ────────────────────

    def _add_field(self, parent, label: str, default: str) -> ctk.CTkEntry:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row, text=label, width=110, anchor="w").pack(side="left")
        entry = ctk.CTkEntry(row, placeholder_text=default)
        entry.pack(side="left", fill="x", expand=True, padx=4)
        return entry

    def _add_combo(self, parent, label: str, values: list, default: str) -> ctk.CTkComboBox:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row, text=label, width=110, anchor="w").pack(side="left")
        combo = ctk.CTkComboBox(row, values=values, width=200)
        combo.set(default)
        combo.pack(side="left", padx=4)
        return combo

    def _browse_input(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select FBX File",
            filetypes=[("FBX files", "*.fbx"), ("All files", "*.*")],
        )
        if path:
            self._input_entry.delete(0, "end")
            self._input_entry.insert(0, path)
            if not self._output_entry.get():
                self._output_entry.delete(0, "end")
                self._output_entry.insert(0, str(Path(path).with_suffix('.vrm')))

    def _browse_output(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            title="Save VRM File", defaultextension=".vrm",
            filetypes=[("VRM files", "*.vrm"), ("All files", "*.*")],
        )
        if path:
            self._output_entry.delete(0, "end")
            self._output_entry.insert(0, path)

    def _log(self, message: str):
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

    def _load_fbx_data(self, filepath: str):
        """Load FBX data (CPU work, safe to call from background thread)."""
        from .fbx_loader import load_fbx
        from .bone_mapping import build_bone_mapping

        fbx_data = load_fbx(filepath)
        bone_names = [b.name for b in fbx_data.bones]
        bone_mapping = build_bone_mapping(bone_names)
        return fbx_data, bone_mapping

    def _setup_renderer(self, fbx_data, bone_mapping):
        """Set up renderer with loaded data (must be called on main thread for OpenGL)."""
        from .renderer import ModelRenderer, OrbitCamera

        self._fbx_data = fbx_data
        self._bone_mapping = bone_mapping

        if self._renderer is None:
            self._renderer = ModelRenderer()
        self._renderer.load_model(self._fbx_data)

        if self._camera is None:
            self._camera = OrbitCamera()
        bbox_min, bbox_max = self._renderer.get_bbox()
        self._camera.fit_to_bounds(bbox_min, bbox_max)

    # ──────────────────── Convert ────────────────────

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
                convert_fbx_to_vrm(
                    input_path, output_path, meta=meta,
                    callback=lambda msg, pct: (self._log(msg), self._set_progress(pct, msg)),
                )
                self._log(f"\nConversion successful!\nOutput: {output_path}")
                self._set_progress(1.0, "Conversion complete!")
            except Exception as e:
                self._log(f"\nERROR: {e}")
                self._log(traceback.format_exc())
                self._set_progress(0, f"Error: {e}")
            finally:
                self.root.after(0, lambda: (
                    setattr(self, '_converting', False),
                    self._convert_btn.configure(state="normal", text="Convert to VRM"),
                ))

        threading.Thread(target=_run, daemon=True).start()

    # ──────────────────── Preview ────────────────────

    def _preview_load_model(self):
        if self._loading_model:
            return
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select FBX File",
            filetypes=[("FBX files", "*.fbx"), ("All files", "*.*")],
        )
        if not path:
            return

        self._loading_model = True
        self._preview_info.configure(text="Loading...")
        self.root.update_idletasks()

        def _do_load():
            try:
                # CPU work in background thread
                fbx_data, bone_mapping = self._load_fbx_data(path)
                # Schedule GPU work on main thread
                self.root.after(0, lambda: self._finish_preview_load(path, fbx_data, bone_mapping))
            except Exception as e:
                logger.error(f"Failed to load model: {e}", exc_info=True)
                self.root.after(0, lambda: (
                    self._preview_info.configure(text=f"Error: {e}"),
                    setattr(self, '_loading_model', False),
                ))

        threading.Thread(target=_do_load, daemon=True).start()

    def _finish_preview_load(self, path, fbx_data, bone_mapping):
        """Finish loading on main thread (OpenGL operations)."""
        try:
            self._setup_renderer(fbx_data, bone_mapping)
            n_meshes = len(self._fbx_data.meshes)
            n_bones = len(self._fbx_data.bones)
            n_verts = sum(len(m.positions) for m in self._fbx_data.meshes)
            self._preview_info.configure(
                text=f"{Path(path).name} | {n_meshes} mesh, {n_bones} bones, {n_verts} verts")
            self._start_preview_loop()
        except Exception as e:
            logger.error(f"Failed to set up renderer: {e}", exc_info=True)
            self._preview_info.configure(text=f"Error: {e}")
        finally:
            self._loading_model = False

    def _preview_reset_view(self):
        if self._camera and self._renderer and self._renderer.has_model:
            bbox_min, bbox_max = self._renderer.get_bbox()
            self._camera.fit_to_bounds(bbox_min, bbox_max)

    def _start_preview_loop(self):
        if self._preview_running:
            return
        self._preview_running = True
        self._preview_tick()

    def _preview_tick(self):
        if not self._preview_running:
            return
        if self._renderer and self._renderer.has_model and self._camera:
            try:
                w = max(self._preview_canvas.winfo_width(), 320)
                h = max(self._preview_canvas.winfo_height(), 240)
                img = self._renderer.render(w, h, self._camera)
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
                self._preview_canvas.configure(image=ctk_img, text="")
                self._preview_canvas._ctk_image = ctk_img  # prevent GC
            except Exception as e:
                logger.warning(f"Preview render error: {e}")

        self._preview_after_id = self.root.after(FRAME_MS, self._preview_tick)

    # ──── Preview Mouse Controls ────

    def _on_preview_press(self, event):
        self._mouse_last_x = event.x
        self._mouse_last_y = event.y
        self._preview_drag_button = event.num
        self._preview_shift = bool(event.state & 0x1)  # Shift key

    def _on_preview_drag(self, event):
        if self._camera is None:
            return
        dx = event.x - self._mouse_last_x
        dy = event.y - self._mouse_last_y
        self._mouse_last_x = event.x
        self._mouse_last_y = event.y

        shift = bool(event.state & 0x1)

        if self._preview_drag_button == 1 and shift:
            # Shift + Left drag: pan
            self._camera.pan(dx, dy)
        elif self._preview_drag_button == 1:
            # Left drag: orbit
            self._camera.orbit(dx, dy)
        elif self._preview_drag_button == 3:
            # Right drag: zoom
            self._camera.zoom(dx, dy)

    def _on_preview_release(self, event):
        self._preview_drag_button = None

    def _on_preview_scroll(self, event):
        if self._camera:
            # Windows: event.delta is ±120
            self._camera.scroll_zoom(event.delta / 120.0)

    def _on_preview_scroll_linux(self, event, direction):
        if self._camera:
            self._camera.scroll_zoom(direction)

    # ──────────────────── MoCap ────────────────────

    def _mocap_load_model(self):
        if self._loading_model:
            return
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select FBX File",
            filetypes=[("FBX files", "*.fbx"), ("All files", "*.*")],
        )
        if not path:
            return
        self._loading_model = True
        self._mocap_status.configure(text="Loading model...")
        self.root.update_idletasks()

        def _do_load():
            try:
                fbx_data, bone_mapping = self._load_fbx_data(path)
                self.root.after(0, lambda: self._finish_mocap_load(path, fbx_data, bone_mapping))
            except Exception as e:
                logger.error(f"Failed to load model for MoCap: {e}", exc_info=True)
                self.root.after(0, lambda: (
                    self._mocap_status.configure(text=f"Error: {e}"),
                    setattr(self, '_loading_model', False),
                ))

        threading.Thread(target=_do_load, daemon=True).start()

    def _finish_mocap_load(self, path, fbx_data, bone_mapping):
        """Finish MoCap model loading on main thread (OpenGL operations)."""
        try:
            self._setup_renderer(fbx_data, bone_mapping)
            self._mocap_status.configure(text=f"Model loaded: {Path(path).name}")
        except Exception as e:
            logger.error(f"Failed to set up renderer: {e}", exc_info=True)
            self._mocap_status.configure(text=f"Error: {e}")
        finally:
            self._loading_model = False

    def _mocap_toggle(self):
        if self._mocap_running:
            self._mocap_stop()
        else:
            self._mocap_start()

    def _mocap_start(self):
        from .motion_capture import MotionCapture
        from .pose_solver import PoseSolver

        if not self._renderer or not self._renderer.has_model:
            self._mocap_status.configure(text="Load a model first!")
            return

        cam_idx = 0
        try:
            cam_idx = int(self._cam_index_entry.get())
        except ValueError:
            pass

        self._mocap_status.configure(text="Starting camera...")
        self.root.update_idletasks()

        try:
            if self._mocap is None:
                self._mocap = MotionCapture()
            self._mocap.mirror = self._mirror_var.get()
            self._mocap.start(camera_index=cam_idx)

            if self._pose_solver is None:
                self._pose_solver = PoseSolver()

            if self._camera is None:
                from .renderer import OrbitCamera
                self._camera = OrbitCamera()
                if self._renderer.has_model:
                    bbox_min, bbox_max = self._renderer.get_bbox()
                    self._camera.fit_to_bounds(bbox_min, bbox_max)

            self._mocap_running = True
            self._mocap_start_btn.configure(
                text="Stop Camera", fg_color="#dc3545", hover_color="#c82333")
            self._mocap_status.configure(text="Capturing...")
            self._mocap_tick()

        except Exception as e:
            logger.error(f"Failed to start MoCap: {e}", exc_info=True)
            self._mocap_status.configure(text=f"Error: {e}")

    def _mocap_stop(self):
        self._mocap_running = False
        if self._mocap_after_id:
            self.root.after_cancel(self._mocap_after_id)
            self._mocap_after_id = None
        if self._mocap:
            self._mocap.stop()
        if self._renderer:
            self._renderer.reset_pose()
        self._mocap_start_btn.configure(
            text="Start Camera", fg_color="#28a745", hover_color="#218838")
        self._mocap_status.configure(text="Camera off")
        self._mocap_fps_label.configure(text="FPS: --")

    def _mocap_update_mirror(self):
        if self._mocap:
            self._mocap.mirror = self._mirror_var.get()

    def _mocap_tick(self):
        if not self._mocap_running:
            return

        try:
            frame, landmarks, world_landmarks = self._mocap.get_latest()

            # Update camera view
            if frame is not None:
                cw = max(self._mocap_cam_label.winfo_width(), 200)
                ch = max(self._mocap_cam_label.winfo_height(), 150)
                cam_img = frame.resize((cw, ch), PILImage.LANCZOS)
                ctk_cam = ctk.CTkImage(light_image=cam_img, dark_image=cam_img, size=(cw, ch))
                self._mocap_cam_label.configure(image=ctk_cam, text="")
                self._mocap_cam_label._ctk_img = ctk_cam

            # Solve pose and update model
            use_landmarks = world_landmarks if world_landmarks else landmarks
            if (use_landmarks and self._pose_solver and self._fbx_data
                    and self._bone_mapping and self._renderer
                    and self._renderer.rest_global is not None):
                from .pose_solver import apply_rotations_to_skeleton

                rotations = self._pose_solver.solve(use_landmarks)
                if rotations:
                    new_global = apply_rotations_to_skeleton(
                        rotations,
                        self._fbx_data.bones,
                        self._fbx_data.bone_name_to_index,
                        self._bone_mapping,
                        self._renderer.rest_global,
                    )
                    self._renderer.set_bone_transforms(new_global)

            # Render model view
            if self._renderer and self._renderer.has_model and self._camera:
                mw = max(self._mocap_model_label.winfo_width(), 200)
                mh = max(self._mocap_model_label.winfo_height(), 150)
                model_img = self._renderer.render(mw, mh, self._camera)
                ctk_model = ctk.CTkImage(light_image=model_img, dark_image=model_img, size=(mw, mh))
                self._mocap_model_label.configure(image=ctk_model, text="")
                self._mocap_model_label._ctk_img = ctk_model

            # Update FPS
            fps = self._mocap.fps if self._mocap else 0
            self._mocap_fps_label.configure(text=f"FPS: {fps:.0f}")

        except Exception as e:
            logger.warning(f"MoCap tick error: {e}")

        self._mocap_after_id = self.root.after(FRAME_MS, self._mocap_tick)

    # ──────────────────── Lifecycle ────────────────────

    def _on_close(self):
        self._preview_running = False
        self._mocap_running = False
        if self._preview_after_id:
            self.root.after_cancel(self._preview_after_id)
        if self._mocap_after_id:
            self.root.after_cancel(self._mocap_after_id)
        if self._mocap:
            self._mocap.stop()
        if self._renderer:
            self._renderer.cleanup()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app = App()
    app.run()


if __name__ == "__main__":
    main()
