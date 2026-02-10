# EZ-FBX2VRM 仕様書

**バージョン:** 1.0
**最終更新:** 2026-02-10
**ドキュメント種別:** ソフトウェア仕様書

---

## 1. 概要

### 1.1 製品名
EZ-FBX2VRM

### 1.2 目的
Tripo 等の 3D 生成ツールで作成し Mixamo でリギングした FBX 形式の 3D モデルを、**Blender や Unity を使わずに** VRM 形式に変換する Windows デスクトップアプリケーション。

### 1.3 主要機能
| # | 機能 | 概要 |
|---|------|------|
| F1 | FBX → VRM 変換 | Mixamo リグ付き FBX を VRM 0.x に変換 |
| F2 | 3D プレビュー | OpenGL でモデルをリアルタイム表示。マウスによるカメラ操作 |
| F3 | モーションキャプチャー | Web カメラ映像から人体ポーズを推定し、3D モデルをリアルタイム駆動 |
| F4 | CLI モード | コマンドラインからのバッチ変換 |
| F5 | .exe ビルド | PyInstaller による単体実行ファイル配布 |

### 1.4 対象プラットフォーム
- Windows 10 / 11 (64-bit)
- Python 3.10 以上

---

## 2. アーキテクチャ

### 2.1 モジュール構成

```
EZ-FBX2VRM/
├── main.py                 # エントリーポイント (GUI/CLI 分岐)
├── src/
│   ├── __init__.py          # パッケージ初期化
│   ├── bone_mapping.py      # Mixamo → VRM ボーンマッピング定義
│   ├── fbx_loader.py        # FBX ファイル読み込み (pyassimp)
│   ├── vrm_builder.py       # VRM (glTF 2.0 + VRM 拡張) 構築・GLB 出力
│   ├── converter.py         # 変換パイプライン (Load → Build → Write)
│   ├── renderer.py          # OpenGL 3D レンダラー (moderngl)
│   ├── pose_solver.py       # MediaPipe ランドマーク → ボーン回転変換
│   ├── motion_capture.py    # Web カメラ + MediaPipe ポーズ推定
│   └── gui.py               # CustomTkinter GUI (3 タブ)
├── assets/                  # アイコン等
├── requirements.txt         # Python 依存パッケージ
├── build.spec               # PyInstaller ビルド設定
├── build.bat                # Windows ビルドスクリプト
└── setup.bat                # Windows 環境セットアップスクリプト
```

### 2.2 データフロー

```
┌──────────────┐
│  FBX ファイル  │  (Mixamo リグ付き)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ fbx_loader   │  pyassimp で読み込み
│              │  → FBXData (meshes, bones, materials)
└──────┬───────┘
       │
       ├─────────────────────┬──────────────────────────┐
       ▼                     ▼                          ▼
┌──────────────┐   ┌─────────────────┐   ┌──────────────────────┐
│ vrm_builder  │   │   renderer      │   │ pose_solver           │
│              │   │                 │   │ + motion_capture      │
│ glTF 2.0 +   │   │ OpenGL 3.3     │   │                      │
│ VRM 拡張構築  │   │ オフスクリーン   │   │ MediaPipe → Quaternion│
│ → GLB binary │   │ → PIL Image    │   │ → Bone Transforms    │
└──────┬───────┘   └────────┬────────┘   └───────────┬──────────┘
       │                    │                        │
       ▼                    ▼                        ▼
┌──────────────┐   ┌──────────────────────────────────┐
│ .vrm ファイル │   │         gui.py (CustomTkinter)    │
│  (出力)      │   │   ┌──────┬──────────┬──────────┐  │
└──────────────┘   │   │Convert│ Preview  │  MoCap   │  │
                   │   └──────┴──────────┴──────────┘  │
                   └──────────────────────────────────┘
```

### 2.3 依存ライブラリ

| ライブラリ | バージョン | 用途 |
|-----------|-----------|------|
| pygltflib | >= 1.16.1 | glTF 2.0 データ構造 |
| numpy | >= 1.24.0 | 行列演算・クォータニオン |
| customtkinter | >= 5.2.0 | モダン GUI フレームワーク |
| pyassimp | >= 4.1.4 | FBX/3D モデル読み込み |
| Pillow | >= 10.0.0 | 画像処理 (テクスチャ・フレーム変換) |
| moderngl | >= 5.8.0 | OpenGL 3.3 オフスクリーンレンダリング |
| opencv-python | >= 4.8.0 | Web カメラキャプチャ |
| mediapipe | >= 0.10.0 | 人体ポーズ推定 (33 ランドマーク) |

**外部依存:** Assimp 共有ライブラリ (.dll) が別途必要

---

## 3. 機能仕様

### 3.1 F1: FBX → VRM 変換

#### 3.1.1 入力
- **形式:** FBX (Binary/ASCII)
- **前提条件:** Mixamo でリギング済み (標準ボーン命名規則)
- **対応プレフィックス:** `mixamorig:`, `mixamorig.`, `mixamorig_`, `mixamorig1:`

#### 3.1.2 出力
- **形式:** VRM 0.x (.vrm = glTF 2.0 Binary + VRM 拡張 JSON)
- **構造:** GLB (12 バイトヘッダ + JSON チャンク + BIN チャンク)

#### 3.1.3 変換パイプライン

| フェーズ | 処理 | 進捗 |
|---------|------|------|
| Phase 1 | FBX 読み込み (pyassimp) | 0% → 40% |
| Phase 2 | VRM 構築 (glTF + VRM 拡張) | 40% → 80% |
| Phase 3 | ファイル書き出し | 80% → 100% |

#### 3.1.4 FBX 読み込み処理 (`fbx_loader.py`)

**pyassimp 処理フラグ:**
| フラグ | 効果 |
|-------|------|
| `aiProcess_Triangulate` | 全ポリゴンを三角形に変換 |
| `aiProcess_GenNormals` | 法線ベクトルの自動生成 |
| `aiProcess_JoinIdenticalVertices` | 重複頂点の統合 |
| `aiProcess_LimitBoneWeights` | ボーンウェイトを頂点あたり最大 4 に制限 |
| `aiProcess_FlipUVs` | UV 座標の Y 軸反転 |

**抽出データ:**

| データ | 型 | 説明 |
|--------|---|------|
| `meshes` | `list[MeshData]` | メッシュ群 (頂点・法線・UV・インデックス・スキニング) |
| `bones` | `list[BoneInfo]` | ボーン階層 (名前・親子関係・変換行列) |
| `materials` | `list[MaterialData]` | マテリアル (色・テクスチャパス・金属度・粗さ) |

**MeshData 構造:**

| フィールド | 型 | 説明 |
|-----------|---|------|
| `positions` | `np.ndarray(N, 3) float32` | 頂点座標 |
| `normals` | `np.ndarray(N, 3) float32` | 法線ベクトル |
| `texcoords` | `np.ndarray(N, 2) float32` | UV 座標 |
| `indices` | `np.ndarray(M,) uint32` | 三角形インデックス |
| `joint_indices` | `np.ndarray(N, 4) uint16` | 頂点あたりボーンインデックス (最大 4) |
| `joint_weights` | `np.ndarray(N, 4) float32` | 頂点あたりボーンウェイト (正規化済) |
| `material_index` | `int` | マテリアル参照 |

**BoneInfo 構造:**

| フィールド | 型 | 説明 |
|-----------|---|------|
| `name` | `str` | ボーン名 |
| `parent_index` | `int` | 親ボーンインデックス (-1 = ルート) |
| `children_indices` | `list[int]` | 子ボーンインデックス群 |
| `offset_matrix` | `np.ndarray(4, 4)` | バインドポーズオフセット行列 |
| `local_transform` | `np.ndarray(4, 4)` | 親相対ローカル変換行列 |
| `global_transform` | `np.ndarray(4, 4)` | ワールド変換行列 |

#### 3.1.5 ボーンマッピング (`bone_mapping.py`)

**必須ボーン (17 本):**

| VRM ボーン名 | Mixamo ボーン名 | 階層 |
|-------------|----------------|------|
| `hips` | `Hips` | ルート |
| `spine` | `Spine` | hips の子 |
| `chest` | `Spine1` | spine の子 |
| `neck` | `Neck` | chest/upperChest の子 |
| `head` | `Head` | neck の子 |
| `leftUpperArm` | `LeftArm` | leftShoulder/chest の子 |
| `leftLowerArm` | `LeftForeArm` | leftUpperArm の子 |
| `leftHand` | `LeftHand` | leftLowerArm の子 |
| `rightUpperArm` | `RightArm` | rightShoulder/chest の子 |
| `rightLowerArm` | `RightForeArm` | rightUpperArm の子 |
| `rightHand` | `RightHand` | rightLowerArm の子 |
| `leftUpperLeg` | `LeftUpLeg` | hips の子 |
| `leftLowerLeg` | `LeftLeg` | leftUpperLeg の子 |
| `leftFoot` | `LeftFoot` | leftLowerLeg の子 |
| `rightUpperLeg` | `RightUpLeg` | hips の子 |
| `rightLowerLeg` | `RightLeg` | rightUpperLeg の子 |
| `rightFoot` | `RightFoot` | rightLowerLeg の子 |

**オプションボーン (最大 45 本):**
- `upperChest` (`Spine2`)
- `leftShoulder` / `rightShoulder`
- `leftToes` / `rightToes`
- 指ボーン: 各手 5 指 × 3 関節 (Proximal/Intermediate/Distal)
  - Thumb, Index, Middle, Ring, Little (Mixamo: Pinky → VRM: Little)
- `jaw`, `leftEye`, `rightEye`

**検証:** `validate_bone_mapping()` で必須 17 ボーンの存在を確認。不足時は警告を出力し処理は続行。

#### 3.1.6 VRM 構築 (`vrm_builder.py`)

**出力ファイル構造 (GLB):**
```
┌─────────────────────────────────────────┐
│ GLB Header (12 bytes)                   │
│   magic: 0x46546C67 ("glTF")           │
│   version: 2                            │
│   total length                          │
├─────────────────────────────────────────┤
│ JSON Chunk                              │
│   chunk length + type (0x4E4F534A)     │
│   glTF JSON + VRM extension            │
├─────────────────────────────────────────┤
│ BIN Chunk                               │
│   chunk length + type (0x004E4942)     │
│   メッシュ・テクスチャ等のバイナリデータ    │
└─────────────────────────────────────────┘
```

**glTF JSON 内 VRM 拡張 (`extensions.VRM`):**

```json
{
  "exporterVersion": "EZ-FBX2VRM-1.0",
  "specVersion": "0.0",
  "meta": {
    "title": "...",
    "author": "...",
    "allowedUserName": "Everyone|OnlyAuthor|ExplicitlyLicensedPerson",
    "violentUssageName": "Allow|Disallow",
    "sexualUssageName": "Allow|Disallow",
    "commercialUssageName": "Allow|Disallow",
    "licenseName": "CC0|CC_BY|..."
  },
  "humanoid": {
    "humanBones": [
      {"bone": "hips", "node": 0, "useDefaultValues": true},
      ...
    ]
  },
  "firstPerson": { ... },
  "blendShapeMaster": { "blendShapeGroups": [] },
  "secondaryAnimation": { "boneGroups": [], "colliderGroups": [] },
  "materialProperties": [ ... ]
}
```

**マテリアル変換:**
- FBX マテリアル → glTF PBR (`pbrMetallicRoughness`)
- `baseColorFactor`: Diffuse カラー (RGBA)
- `metallicFactor`: 金属度 (デフォルト 0.0)
- `roughnessFactor`: 粗さ (デフォルト 0.9)
- テクスチャ: PNG に変換して GLB バイナリに埋め込み
- `doubleSided: true`
- VRM マテリアル: `VRM_USE_GLTFSHADER` シェーダー

**行列分解:** 4x4 変換行列 → Translation (vec3) + Rotation (quaternion xyzw) + Scale (vec3) に分解して glTF ノードに格納。

#### 3.1.7 VRM メタデータ

GUI で設定可能な項目:

| 項目 | 型 | 選択肢 | デフォルト |
|------|---|--------|----------|
| Title | テキスト | 自由入力 | "My Model" |
| Author | テキスト | 自由入力 | "Unknown" |
| Version | テキスト | 自由入力 | "1.0" |
| License | 選択 | CC0, CC_BY, CC_BY_NC, CC_BY_SA, CC_BY_NC_SA, CC_BY_ND, CC_BY_NC_ND, Redistribution_Prohibited, Other | CC0 |
| Allowed User | 選択 | OnlyAuthor, ExplicitlyLicensedPerson, Everyone | Everyone |
| Violent Usage | 選択 | Allow, Disallow | Disallow |
| Sexual Usage | 選択 | Allow, Disallow | Disallow |
| Commercial Usage | 選択 | Allow, Disallow | Disallow |

---

### 3.2 F2: 3D プレビュー

#### 3.2.1 レンダリングエンジン (`renderer.py`)

| 項目 | 仕様 |
|------|------|
| API | OpenGL 3.3 Core Profile |
| ライブラリ | moderngl (オフスクリーンスタンドアロンコンテキスト) |
| シェーディング | Blinn-Phong (拡散 + 鏡面 + 環境光) |
| スキニング | GPU 頂点シェーダー (最大 128 ボーン) |
| 出力 | PIL RGBA Image → CTkImage で GUI に表示 |
| フレームレート | 30 FPS (33ms 間隔) |
| 背景色 | (0.18, 0.18, 0.20) ダークグレー |

**シェーダー仕様:**

**頂点シェーダー:**
- 入力: `position(vec3)`, `normal(vec3)`, `texcoord(vec2)`, `joints(vec4)`, `weights(vec4)`
- ユニフォーム: `model/view/projection(mat4)`, `bone_matrices[128](mat4)`, `use_skinning(int)`
- スキニング: `skinned_pos = Σ(weight[i] × bone_matrix[joint[i]] × position)` (最大 4 ボーン)
- 出力: `world_pos(vec3)`, `normal(vec3)`, `texcoord(vec2)`

**フラグメントシェーダー:**
- 入力: ワールド座標、法線、UV
- ライティング: `diffuse = max(N·L, 0)`, `specular = pow(max(N·H, 0), 32) × 0.3`
- ユニフォーム: `light_dir(0.5, 1.0, 0.8)`, `light_color(0.9, 0.88, 0.85)`, `ambient(0.25, 0.25, 0.3)`
- テクスチャサンプリング (オプション)

**グリッド:**
- XZ 平面、±10 範囲、0.5 刻み
- X 軸: 赤系 (0.6, 0.2, 0.2)、Z 軸: 青系 (0.2, 0.2, 0.6)、その他: グレー

#### 3.2.2 カメラ操作 (`OrbitCamera`)

| 操作 | 入力 | 動作 | パラメータ |
|------|------|------|----------|
| 回転 (Orbit) | 左ドラッグ | Yaw / Pitch を変更 | 感度: ×0.5, Pitch 制限: ±89° |
| ズーム | 右ドラッグ (垂直) | Distance を変更 | 係数: 1 + dy×0.005, 範囲: 0.1 〜 50.0 |
| パン | Shift + 左ドラッグ | Target 位置を移動 | 速度: distance × 0.002 |
| スクロールズーム | マウスホイール | Distance を変更 | 係数: 1 - delta×0.1, 範囲: 0.1 〜 50.0 |
| フィット | "Reset View" ボタン | モデル境界ボックスに合わせる | distance = size × 1.5 |

**カメラ初期値:**
- Target: (0.0, 0.85, 0.0)
- Distance: 3.0
- Yaw: 0°, Pitch: 15°
- FOV: 45°, Near: 0.01, Far: 100.0

**座標系:** 右手系 Y-up (glTF/VRM 標準と同一)

---

### 3.3 F3: モーションキャプチャー

#### 3.3.1 カメラキャプチャ (`motion_capture.py`)

| 項目 | 仕様 |
|------|------|
| ライブラリ | OpenCV (`cv2.VideoCapture`) |
| デフォルト解像度 | 640 × 480 |
| フレームレート上限 | 30 FPS |
| カメラ自動検出 | インデックス 0〜4 をスキャン |
| ミラーモード | デフォルト ON (`cv2.flip(frame, 1)`) |
| スレッド | バックグラウンドスレッド (daemon) で連続キャプチャ |
| FPS 計測 | 直近 30 フレームの移動平均 |

#### 3.3.2 ポーズ推定 (MediaPipe Pose)

| 項目 | 仕様 |
|------|------|
| モデル | MediaPipe Pose (model_complexity=1) |
| ランドマーク数 | 33 点 (全身) |
| 出力 | 正規化座標 (画面 0〜1) + ワールド座標 (メートル単位) |
| smooth_landmarks | 有効 |
| min_detection_confidence | 0.5 |
| min_tracking_confidence | 0.5 |
| 骨格オーバーレイ | カメラ映像上に描画 |

**使用するランドマーク (33 点):**

| ID | 名前 | 用途 |
|----|------|------|
| 0 | Nose | 頭部方向 |
| 7, 8 | Left/Right Ear | 頭部位置 |
| 9, 10 | Mouth Left/Right | 頭部方向補助 |
| 11, 12 | Left/Right Shoulder | 肩・体幹方向 |
| 13, 14 | Left/Right Elbow | 上腕方向 |
| 15, 16 | Left/Right Wrist | 前腕方向 |
| 23, 24 | Left/Right Hip | 腰位置・体幹方向 |
| 25, 26 | Left/Right Knee | 太もも方向 |
| 27, 28 | Left/Right Ankle | 脛方向 |

#### 3.3.3 ポーズ解決 (`pose_solver.py`)

**座標変換:**
- MediaPipe: x=右, y=下, z=カメラ方向
- 変換後: x=右, y=上, z=前方

**解決対象ボーン:**

| VRM ボーン | 算出方法 |
|-----------|---------|
| `spine` | 腰中心 → 肩中心 方向 |
| `chest` | 腰中心 → 肩中心 方向 |
| `neck` | 肩中心 → 耳中点 方向 |
| `head` | 鼻 → 口中点の上方向 |
| `leftUpperArm` | 左肩 → 左肘 方向 |
| `leftLowerArm` | 左肘 → 左手首 方向 (上腕ローカル空間) |
| `rightUpperArm` | 右肩 → 右肘 方向 |
| `rightLowerArm` | 右肘 → 右手首 方向 (上腕ローカル空間) |
| `leftUpperLeg` | 左腰 → 左膝 方向 |
| `leftLowerLeg` | 左膝 → 左足首 方向 (太ももローカル空間) |
| `rightUpperLeg` | 右腰 → 右膝 方向 |
| `rightLowerLeg` | 右膝 → 右足首 方向 (太ももローカル空間) |

**回転計算アルゴリズム:**
1. ランドマーク位置から各ボーンの方向ベクトルを算出
2. T ポーズ基準方向 (`REST_DIRECTIONS`) との差分を求める
3. `_quat_from_two_vectors()` で基準→現在のクォータニオンを算出
4. 子ボーンは親の逆回転を適用してローカル空間で計算
5. SLERP (`smoothing=0.4`) で前フレームとの時間平滑化

**T ポーズ基準方向:**
| ボーン群 | 方向 |
|---------|------|
| spine, chest, upperChest, neck, head | (0, 1, 0) 上 |
| leftUpperArm, leftLowerArm | (1, 0, 0) 右 |
| rightUpperArm, rightLowerArm | (-1, 0, 0) 左 |
| leftUpperLeg, leftLowerLeg | (0, -1, 0) 下 |
| rightUpperLeg, rightLowerLeg | (0, -1, 0) 下 |

**クォータニオン演算ユーティリティ:**
- `_quat_from_two_vectors` - 2 ベクトル間回転
- `_quat_multiply` - 乗算
- `_quat_inverse` - 逆 (共役)
- `_quat_to_mat4` - 4x4 行列変換
- `_slerp` - 球面線形補間
- `_rotate_vec_by_quat` - ベクトル回転

#### 3.3.4 MoCap タブ UI レイアウト

```
┌─────────────────────────────────────────────────┐
│ [Load FBX] [Start Camera] Camera:[0] [✓ Mirror] │
│                              Camera off  FPS: -- │
├────────────────────────┬────────────────────────┤
│       Camera           │        Model           │
│                        │                        │
│  (Web カメラ映像 +      │  (OpenGL 3D レンダリング  │
│   骨格オーバーレイ)     │   ポーズ連動)           │
│                        │                        │
└────────────────────────┴────────────────────────┘
```

---

### 3.4 F4: CLI モード

**起動コマンド:**
```bash
python main.py --cli <input.fbx> [options]
```

**引数:**

| 引数 | 短縮 | 必須 | 説明 | デフォルト |
|------|------|------|------|----------|
| `input` | - | はい | 入力 FBX ファイルパス | - |
| `--output` | `-o` | いいえ | 出力 VRM ファイルパス | 入力名.vrm |
| `--title` | - | いいえ | モデルタイトル | "My Model" |
| `--author` | - | いいえ | 作者名 | "Unknown" |
| `--license` | - | いいえ | ライセンス種別 | "CC0" |
| `--verbose` | `-v` | いいえ | 詳細ログ出力 | OFF |

**進捗表示:**
```
[############------------------]  40.0% Phase 1/3: Loading FBX...
```

---

## 4. GUI 仕様

### 4.1 ウィンドウ

| 項目 | 値 |
|------|---|
| タイトル | "EZ-FBX2VRM" |
| 初期サイズ | 1000 × 750 px |
| 最小サイズ | 800 × 650 px |
| テーマ | ダークモード |
| カラーテーマ | Blue |
| フレームワーク | CustomTkinter |

### 4.2 タブ構成

#### タブ 1: Convert

```
┌──────────────────────────────────────────┐
│ Input FBX:  [____________________] [Browse] │
│ Output VRM: [____________________] [Browse] │
├──────────────────────────────────────────┤
│ VRM Metadata                             │
│   Title:    [____________________]       │
│   Author:   [____________________]       │
│   Version:  [____________________]       │
│   License:          [CC0         ▼]     │
│   Allowed User:     [Everyone    ▼]     │
│   Violent Usage:    [Disallow    ▼]     │
│   Sexual Usage:     [Disallow    ▼]     │
│   Commercial:       [Disallow    ▼]     │
├──────────────────────────────────────────┤
│         [ Convert to VRM ]               │
│ [════════════════════════════] 0%         │
│ Ready                                    │
├──────────────────────────────────────────┤
│ (ログ出力エリア)                          │
└──────────────────────────────────────────┘
```

#### タブ 2: Preview

```
┌──────────────────────────────────────────────┐
│ [Load FBX] [Reset View]     model info label │
│ Left drag: Rotate | Right drag: Zoom | ...   │
├──────────────────────────────────────────────┤
│                                              │
│          (3D ビューポート)                     │
│          OpenGL レンダリング画像               │
│          マウスイベント受付                    │
│                                              │
└──────────────────────────────────────────────┘
```

#### タブ 3: MoCap

```
┌────────────────────────────────────────────────┐
│ [Load FBX] [Start Camera] Camera:[0] [✓Mirror] │
│                              status    FPS: -- │
├───────────────────────┬────────────────────────┤
│       Camera          │       Model            │
│   (Web カメラ映像)     │   (3D レンダリング)     │
└───────────────────────┴────────────────────────┘
```

### 4.3 スレッドモデル

| スレッド | 用途 | 更新間隔 |
|---------|------|---------|
| メインスレッド | tkinter イベントループ | - |
| 変換スレッド | FBX → VRM 変換処理 | (バッチ) |
| FBX 読み込みスレッド | モデルロード | (都度) |
| カメラキャプチャスレッド | Web カメラ + MediaPipe 処理 | ~30 FPS |
| Preview 描画ループ | `root.after(33ms)` でメインスレッド上 | 30 FPS |
| MoCap 描画ループ | `root.after(33ms)` でメインスレッド上 | 30 FPS |

**スレッド安全性:**
- UI 更新は全て `root.after(0, callback)` 経由
- カメラフレーム・ランドマークは `threading.Lock` で保護
- バックグラウンドスレッドは全て `daemon=True`

---

## 5. ビルド仕様

### 5.1 PyInstaller 設定

| 項目 | 値 |
|------|---|
| 出力名 | `EZ-FBX2VRM.exe` |
| コンソール | 非表示 (`console=False`) |
| UPX 圧縮 | 有効 |
| 同梱データ | `assets/` ディレクトリ |
| Hidden Imports | pygltflib, numpy, customtkinter, pyassimp, PIL, moderngl, cv2, mediapipe |

### 5.2 ビルド手順

```bash
pip install -r requirements.txt
pip install pyinstaller
pyinstaller build.spec --clean
# → dist/EZ-FBX2VRM.exe
```

---

## 6. 制約・制限事項

| # | 項目 | 内容 |
|---|------|------|
| L1 | VRM バージョン | 0.x のみ対応 (1.0 未対応) |
| L2 | 入力ボーン | Mixamo 命名規則準拠のみ自動マッピング |
| L3 | スキニング | 頂点あたり最大 4 ボーン |
| L4 | レンダラーボーン数 | GPU シェーダー上限 128 本 |
| L5 | ブレンドシェイプ | 未対応 (空配列で出力) |
| L6 | Spring Bone | 未対応 (空配列で出力) |
| L7 | MToon シェーダー | 未対応 (PBR/glTF シェーダーのみ) |
| L8 | アニメーション | FBX 内アニメーションは変換対象外 |
| L9 | MoCap 指トラッキング | 未対応 (体幹・四肢のみ) |
| L10 | MoCap 表情トラッキング | 未対応 |
| L11 | OpenGL | 3.3 以上必須 (GPU ドライバ依存) |

---

## 7. コード統計

| 指標 | 値 |
|------|---|
| 総ソースファイル数 | 10 (.py) |
| 総コード行数 | 約 3,150 行 |
| クラス数 | 9 (データクラス 4 + ロジック 5) |
| 関数数 | 約 65 |
| 外部依存パッケージ数 | 8 |

---

## 8. 将来的な拡張候補

| # | 機能 | 概要 |
|---|------|------|
| E1 | VRM 1.0 対応 | VRMC_* モジュラー拡張、+Z フォワード |
| E2 | ブレンドシェイプ | 表情 (Joy, Angry, Sorrow, Fun, Surprise) |
| E3 | Spring Bone | 髪・衣服の物理揺れ設定 |
| E4 | MToon マテリアル | トゥーンシェーディング対応 |
| E5 | 指トラッキング | MediaPipe Hands による指の MoCap |
| E6 | 表情トラッキング | MediaPipe Face Mesh による表情 MoCap |
| E7 | カスタムボーンマッピング | Mixamo 以外のリグへの対応 (GUI で手動マッピング) |
| E8 | アニメーション変換 | FBX アニメーション → VRM アニメーション |
| E9 | 3D プレビューテクスチャ | プレビューでのテクスチャ表示対応 |
| E10 | MoCap 録画 | モーションキャプチャーの録画・再生・エクスポート |
