from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from PIL import Image
import io

# 1. 初始化 FastAPI 框架和模板引擎
app = FastAPI(title="芋头 AI 质检站 API")
templates = Jinja2Templates(directory="templates")

# 2. 唤醒你的 AI 大脑
# 记得替换成你自己的 best.pt 路径！！！
model_path = "C:/Users/Lenovo--/Desktop/taro_dataset/runs/detect/train10/weights/best.pt" 
model = YOLO(model_path)
print("✅ FastAPI 引擎启动！模型加载成功！")

# 3. 首页路由：当用户访问网址时，返回 taro.html 页面
@app.get("/")
async def home(request: Request):
    # FastAPI 规定返回模板时，必须带上 request 这个参数
    return templates.TemplateResponse("taro.html", {"request": request})

# 4. 预测接口：专门接收前端传来的多张图片
# 4. 预测接口：专门接收前端传来的多张图片
# 4. 预测接口：终极调试版
# 4. 预测接口：终极调试版
@app.post("/predict")
async def predict(images: list[UploadFile] = File(...)):
    total_normal = 0
    total_diseased = 0
    total_unrecognized = 0 
    results_list = []

    for file in images:
        contents = await file.read()
        
        # 读取图片，先不强制转RGB，保留原始状态试试
        img = Image.open(io.BytesIO(contents))
        
        # 🌟 关键 1：加上 conf=0.2，降低置信度阈值，让模型“宁可错杀不放过”
        results = model(img, conf=0.2)
        
        file_normal = 0
        file_diseased = 0
        
        # 统计画框数量
        for r in results:
            # 🌟 关键 2：把模型看完的图片（带框的）保存在你的项目文件夹里！
            # 这样你就能亲眼看到前端传过来的到底是什么鬼样子了
            save_path = f"debug_output_{file.filename}"
            r.save(filename=save_path)
            print(f"📸 调试图片已保存至: {save_path}")

            for c in r.boxes.cls:
                class_name = model.names[int(c)]
                if class_name == 'Normal':
                    file_normal += 1
                    total_normal += 1
                elif class_name == 'Diseased':
                    file_diseased += 1
                    total_diseased += 1
        
        # 判定
        if file_diseased > 0:
            status = 'Diseased'
        elif file_normal > 0:
            status = 'Normal'
        else:
            status = 'Unrecognized' 
            total_unrecognized += 1

        results_list.append({'filename': file.filename, 'status': status})

    return {
        'normal_count': total_normal,
        'diseased_count': total_diseased,
        'unrecognized_count': total_unrecognized,
        'details': results_list
    }