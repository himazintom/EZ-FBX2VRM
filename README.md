# EZ-FBX2VRM

Mixamo でリギングした FBX ファイルを **Blender や Unity を使わずに** VRM 形式に変換する Windows デスクトップアプリケーションです。

## 概要

Tripo などの 3D 生成ツールで作成し、Mixamo でリギングしたモデルを、ワンクリックで VRM (VRM 0.x) に変換できます。

**主な特徴:**
- Blender / Unity 不要
- Mixamo ボーン名を自動で VRM ヒューマノイドボーンにマッピング
- VRM メタデータ（作者名、ライセンス等）を GUI で設定可能
- テクスチャ自動埋め込み
- 3D プレビュー (OpenGL) でモデルをリアルタイム確認
- カメラ モーションキャプチャー (MediaPipe) でモデルを動かせる
- GUI モードと CLI モードの両方に対応
- PyInstaller で単体 `.exe` にビルド可能

## 必要環境

- Python 3.10 以上
- Assimp 共有ライブラリ（FBX 読み込みに必要）
- Web カメラ（モーションキャプチャー機能を使う場合）

## セットアップ

### 1. Python パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. Assimp ライブラリのインストール

pyassimp は Assimp の共有ライブラリ (.dll / .so) が必要です。以下のいずれかの方法でインストールしてください。

**方法 A (conda):**
```bash
conda install -c conda-forge assimp
```

**方法 B (vcpkg):**
```bash
vcpkg install assimp:x64-windows
```

**方法 C (手動):**
[Assimp Releases](https://github.com/assimp/assimp/releases) からダウンロードし、`assimp.dll` をプロジェクトフォルダまたは PATH に配置。

### Windows ユーザー向け簡単セットアップ

```
setup.bat
```

## 使い方

### GUI モード

```bash
python main.py
```

アプリケーションは3つのタブで構成されています。

#### Convert タブ - FBX → VRM 変換

1. 「Browse」で入力 FBX ファイルを選択
2. 出力先 VRM パスを確認（自動生成されます）
3. VRM メタデータ（タイトル、作者、ライセンス等）を入力
4. 「Convert to VRM」をクリック

#### Preview タブ - 3D モデルプレビュー

1. 「Load FBX」でモデルを読み込み
2. マウスでモデルを回転・ズーム・移動

**カメラ操作:**
| 操作 | 動作 |
|------|------|
| 左ドラッグ | 回転（オービット） |
| 右ドラッグ | ズームイン/アウト |
| Shift + 左ドラッグ | 水平/垂直移動（パン） |
| マウスホイール | ズームイン/アウト |

#### MoCap タブ - モーションキャプチャー

カメラに写っている人の動きをリアルタイムで 3D モデルに反映させます。

1. 「Load FBX」でモデルを読み込み
2. カメラ番号を選択（デフォルト: 0）
3. 「Start Camera」をクリック
4. カメラの前で動くと、モデルが同期して動きます

**機能:**
- MediaPipe Pose による全身33ポイントのトラッキング
- リアルタイムボーンローテーション解決
- スムージング付きでなめらかな動き
- ミラーモード切り替え可能
- 左側にカメラ映像（スケルトンオーバーレイ付き）、右側に3Dモデル

### CLI モード

```bash
python main.py --cli input.fbx -o output.vrm --title "My Avatar" --author "Author Name"
```

**オプション:**
| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--cli` | CLI モードで実行 | (GUI モード) |
| `-o, --output` | 出力ファイルパス | 入力ファイル名.vrm |
| `--title` | モデルのタイトル | My Model |
| `--author` | 作者名 | Unknown |
| `--license` | ライセンス (CC0, CC_BY, 等) | CC0 |
| `-v, --verbose` | 詳細ログ出力 | off |

## .exe ビルド（Windows 配布用）

```bash
pip install pyinstaller
pyinstaller build.spec --clean
```

`dist/EZ-FBX2VRM.exe` が生成されます。

## 対応フォーマット

- **入力:** FBX (Mixamo リグ付き)
- **出力:** VRM 0.x (.vrm = glTF 2.0 Binary + VRM 拡張)

## ボーンマッピング

Mixamo の標準ボーン名を自動的に VRM ヒューマノイドボーンにマッピングします。

| Mixamo | VRM |
|--------|-----|
| Hips | hips |
| Spine / Spine1 / Spine2 | spine / chest / upperChest |
| Neck / Head | neck / head |
| LeftArm / LeftForeArm | leftUpperArm / leftLowerArm |
| LeftUpLeg / LeftLeg | leftUpperLeg / leftLowerLeg |
| LeftHandThumb1-3 | leftThumbProximal-Distal |
| LeftHandPinky1-3 | leftLittleProximal-Distal |
| (右側も同様) | |

## プロジェクト構成

```
EZ-FBX2VRM/
├── main.py                # エントリーポイント (GUI/CLI)
├── src/
│   ├── __init__.py
│   ├── bone_mapping.py    # Mixamo → VRM ボーンマッピング
│   ├── fbx_loader.py      # FBX ファイルローダー (pyassimp)
│   ├── vrm_builder.py     # VRM (glTF+拡張) ビルダー
│   ├── converter.py       # 変換パイプライン
│   ├── renderer.py        # OpenGL 3D レンダラー (moderngl)
│   ├── pose_solver.py     # MediaPipe → ボーン回転変換
│   ├── motion_capture.py  # ウェブカメラ + MediaPipe キャプチャー
│   └── gui.py             # CustomTkinter GUI (3タブ)
├── assets/                # アイコン等のアセット
├── requirements.txt       # Python 依存パッケージ
├── build.spec             # PyInstaller ビルド設定
├── build.bat              # Windows ビルドスクリプト
└── setup.bat              # Windows セットアップスクリプト
```

## 技術スタック

- **FBX 読み込み:** pyassimp (Open Asset Import Library)
- **VRM 出力:** pygltflib + 手動 VRM 拡張 JSON 構築
- **3D レンダリング:** moderngl (OpenGL 3.3 オフスクリーンレンダリング)
- **モーションキャプチャー:** MediaPipe Pose + OpenCV
- **ポーズ解決:** MediaPipe ランドマーク → VRM ボーン回転 (クォータニオン)
- **GUI:** CustomTkinter (モダンな tkinter ラッパー)
- **数値計算:** NumPy
- **テクスチャ処理:** Pillow
- **ビルド:** PyInstaller

## ライセンス

MIT License
