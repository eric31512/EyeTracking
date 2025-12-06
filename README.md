# EyeTracking - 眼動控制滑鼠

使用眼睛注視來控制滑鼠游標，並透過眨眼來點擊！

## 功能特色

- **眼動追蹤**：使用 MediaPipe Face Mesh 追蹤眼睛位置來移動滑鼠
- **眨眼點擊**：連續眨眼兩次觸發滑鼠左鍵點擊
- **校準系統**：四點校準確保精確的游標控制
- **平滑移動**：使用 EMA 和移動平均演算法，讓游標移動更流暢

## 安裝需求

### 系統需求
- Python 3.7+
- 網路攝影機

### 安裝套件

```bash
pip install -r requirements.txt
```

或手動安裝：

```bash
pip install opencv-python mediapipe pyautogui numpy
```

## 使用方式

```bash
python main.py
```

### 操作說明

| 按鍵 | 功能 |
|------|------|
| `C` | 在校準階段擷取校準點 |
| `Q` | 隨時退出程式 |
| 連續眨眼兩次 | 觸發滑鼠左鍵點擊 |

### 使用流程

1. **校準階段**：程式啟動後會依序顯示四個校準點（左上、右上、左下、右下），請注視螢幕上的校準點並按下 `C` 鍵擷取
2. **控制階段**：校準完成後，即可用眼睛控制滑鼠移動，連續眨眼兩次即可點擊

## 專案結構

```
EyeTracking/
├── main.py          # 程式進入點
├── controller.py    # 眼動滑鼠控制器主類別
├── config.py        # 設定參數
├── landmarks.py     # MediaPipe 臉部特徵點定義與 EAR 計算
├── utils.py         # 平滑器與校準資料類別
└── requirements.txt # 套件依賴
```

## 設定參數

可在 `config.py` 中調整以下參數：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `EMA_ALPHA` | 0.3 | 平滑係數，越低越平滑但延遲越高 |
| `SMOOTHING_WINDOW` | 5 | 移動平均視窗大小 |
| `EAR_THRESHOLD` | 0.21 | 眨眼偵測閾值 |
| `DOUBLE_BLINK_TIME_WINDOW` | 0.8 | 雙眨眼時間視窗（秒） |
| `CLICK_COOLDOWN` | 0.5 | 點擊冷卻時間（秒） |

## 授權

此專案為期末專案，歡迎自由使用與修改。
