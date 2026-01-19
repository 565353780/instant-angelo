"""Time utilities"""

from datetime import datetime


def getCurrentTime():
    """获取当前时间字符串"""
    return datetime.now().strftime('%Y%m%d-%H%M%S')


def getTimestamp():
    """获取时间戳"""
    return datetime.now().timestamp()
