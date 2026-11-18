# WISA 缓存数据浏览器

本地网页查看 WISA 视频片段、原始文本、旧 Qwen 文本和 Qwen3.8 文本。

## 覆盖范围

- `encoded/base/manifest.json` 的全部 **3857** 条 WISA 记录
- 当前 qwen38 400-step 训练 overlay 的 **1600** 条 Qwen3.8 记录
- 旧 Qwen 缓存中实际存在的 text/null 条件（当前约 3328 / 399）
- 页面默认只显示 Qwen3.8 已配对的 1600 条，可切换到全部 3857 条
- 旧 Qwen 或 Qwen3.8 缺失时会明确显示“缺失”，不会用另一版本冒充

## 视频裁剪

按 `encoded/base/manifest.json` 的同一份几何信息裁剪：

- `select` 只选择缓存的 81 个连续源帧
- `scale=resize_w:resize_h:flags=bilinear`
- `crop=w:h:crop_left:crop_top`
- 输出 832×480，帧率来自 `pts` 的实际中位帧间隔

视频是**按需生成**的，首次打开某条样本通常需要 0.5–2 秒；之后从
`media/videos/` 缓存读取。不会一次性生成 3857 个视频。

## 启动

```bash
cd /home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/wisa_cache_viewer
./start.sh
```

然后打开：

```text
http://127.0.0.1:8765/
```

默认只监听 loopback。如果浏览器不在同一台机器上，使用 SSH 端口转发：

```bash
ssh -L 8765:127.0.0.1:8765 <user>@<node1>
```

然后在本机打开上面的地址。不要把 `--host 0.0.0.0` 暴露到集群网络，除非明确需要。

## 页面功能

- 按 ID / caption / 原始文本搜索
- 切换 Qwen3.8 1600 条 / 全部 3857 条
- 切换 train / validation
- 切换正例 `text` / 空条件 `null`
- 左侧分页列表，左侧/右侧方向键翻页
- 视频、缓存首帧、原始 caption、manifest caption
- 旧 Qwen：caption、relations、analysis、prompt、model、缓存来源
- Qwen3.8：caption、relations、analysis、trace.raw、prompt、usage、request id、来源
- 来源视频路径、起始/结束帧、PTS、裁剪几何

## 文件

```text
build_catalog.py      生成 data/catalog.json
server.py             本地 HTTP 服务 + 按需 ffmpeg 裁剪
static/index.html
static/style.css
static/app.js
data/catalog.json     索引（不包含视频和 tensor）
media/videos/         按需生成的裁剪视频缓存
```

`start.sh` 默认使用 `physvid` 环境，因为读取旧 Qwen `.pt` 需要 PyTorch。
