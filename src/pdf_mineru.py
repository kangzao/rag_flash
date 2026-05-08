"""
pdf_mineru.py - MinerU PDF转Markdown API封装

MinerU是合合信息提供的文档解析服务，将PDF转换为Markdown格式。
本模块封装了MinerU云端API的调用流程：
1. 提交PDF转换任务获取task_id
2. 轮询任务状态直到完成
3. 下载并解压转换结果（full.md + 图片资源）

环境变量:
    MINERU_API_KEY: MinerU API密钥（必填）

API文档：https://mineru.net/docs
"""

import requests
import time
import zipfile

# MinerU API密钥（建议通过环境变量 MINERU_API_KEY 设置）
import os

MINERU_API_KEY = os.environ.get('MINERU_API_KEY')
if not MINERU_API_KEY:
    raise ValueError("环境变量 MINERU_API_KEY 未设置")


def get_task_id(file_name: str) -> str:
    """提交PDF转换任务，获取异步任务ID。

    Args:
        file_name: PDF文件名（需存在于oss路径），如'【财报】中芯国际2024年年度报告.pdf'

    Returns:
        str: 任务ID，用于后续查询任务状态和下载结果

    API调用:
        POST https://mineru.net/api/v4/extract/task
        Body: {"url": "oss路径", "is_ocr": True, "enable_formula": False}
    """
    url = 'https://mineru.net/api/v4/extract/task'
    header = {
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {MINERU_API_KEY}"
    }
    # PDF文件在OSS上的公开访问路径
    pdf_url = 'https://vl-image.oss-cn-shanghai.aliyuncs.com/pdf/' + file_name
    data = {
        'url': pdf_url,
        'is_ocr': True,  # 启用OCR识别扫描件
        'enable_formula': False,  # 禁用公式渲染（LaTeX）
    }

    res = requests.post(url, headers=header, json=data)
    print(res.status_code)
    print(res.json())
    print(res.json()["data"])
    task_id = res.json()["data"]['task_id']
    return task_id


def get_result(task_id: str) -> None:
    """轮询任务状态，完成后自动下载并解压结果。

    Args:
        task_id: MinerU任务ID（由get_task_id返回）

    处理流程:
        1. 每5秒轮询任务状态
        2. pending/running: 继续等待
        3. done: 下载full.zip并解压
        4. err_msg非空: 输出错误信息

    输出:
        生成 {task_id}.zip 压缩包和 {task_id}/ 解压目录
        目录中包含 full.md（完整Markdown内容）和图片资源
    """
    url = f'https://mineru.net/api/v4/extract/task/{task_id}'
    header = {
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {MINERU_API_KEY}"
    }

    while True:
        res = requests.get(url, headers=header)
        result = res.json()["data"]
        print(result)
        state = result.get('state')
        err_msg = result.get('err_msg', '')
        # 任务进行中，等待后重试
        if state in ['pending', 'running']:
            print("任务未完成，等待5秒后重试...")
            time.sleep(5)
            continue
        # 任务出错
        if err_msg:
            print(f"任务出错: {err_msg}")
            return
        # 任务完成，下载文件
        if state == 'done':
            full_zip_url = result.get('full_zip_url')
            if full_zip_url:
                local_filename = f"{task_id}.zip"
                print(f"开始下载: {full_zip_url}")
                r = requests.get(full_zip_url, stream=True)
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"下载完成，已保存到: {local_filename}")
                # 下载完成后自动解压
                unzip_file(local_filename)
            else:
                print("未找到 full_zip_url，无法下载。")
            return
        # 未知状态
        print(f"未知状态: {state}")
        return


def unzip_file(zip_path: str, extract_dir: str = None) -> None:
    """解压zip文件到指定目录。

    Args:
        zip_path: zip文件路径
        extract_dir: 解压目标目录，默认为zip文件名（不含.zip后缀）

    输出:
        在extract_dir目录下生成解压后的文件
    """
    if extract_dir is None:
        extract_dir = zip_path.rstrip('.zip')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"已解压到: {extract_dir}")


if __name__ == "__main__":
    file_name = '【财报】中芯国际：中芯国际2024年年度报告.pdf'
    task_id = get_task_id(file_name)
    print('task_id:', task_id)
    get_result(task_id)
