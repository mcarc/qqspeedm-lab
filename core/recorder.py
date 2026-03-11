
import subprocess
import console_ctrl
import sys
import threading

class Recorder:
    def __init__(self, scrcpy_exe_path: str):
        """
        初始化录屏器
        :param scrcpy_exe_path: scrcpy.exe 的绝对或相对路径
        """
        self.scrcpy_path = scrcpy_exe_path

    def _read_stdout(self):
        """
        后台线程函数：实时读取子进程的输出并打印到父进程
        """
        if self._process is None or self._process.stdout is None:
            return
            
        # 逐行读取子进程的输出，直到进程结束
        for line in self._process.stdout:
            
            # 直接输出到父进程的控制台，去掉多余的换行符
            sys.stdout.write(f"[scrcpy] {line.decode('utf-8', errors='ignore')}")
            sys.stdout.flush()

    def start_recording(self, output_filepath: str, bitrate: str, max_size: str, video_buffer: str) -> subprocess.Popen:
        """
        启动 scrcpy 进行录屏
        """
        cmd = [
            self.scrcpy_path,
            "-b", str(bitrate),
            "-m", str(max_size),
            "--video-buffer", str(video_buffer),
            "--video-encoder=OMX.qcom.video.encoder.avc",
            "--print-fps",
            "--record", output_filepath,
            "--no-playback"  # 如果你不需要在电脑上看到实时画面，可以取消注释此行以节省性能
        ]
        
        # 在 Windows 上，我们需要将其放在一个新的进程组中，以便稍后可以发送 CTRL_BREAK_EVENT
        if sys.platform == "win32":
            process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,       # 必须用 PIPE 主动捕获
            stderr=subprocess.STDOUT,     # 关键：将 stderr 的日志合并到 stdout
            # stdin=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE,

            shell=True
            )
            self._process = process  # 保存进程对象以供后续线程使用
        else:
            process = subprocess.Popen(cmd)
            
        self._output_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._output_thread.start()

        return process

    def stop_recording(self, process: subprocess.Popen):
        """
        优雅地停止录屏，确保视频文件正常保存
        """
        if process is None or process.poll() is not None:
            return

        try:
            if sys.platform == "win32":
                # Windows：发送 CTRL_C 信号，让 scrcpy 优雅地封装 mp4 文件
                console_ctrl.send_ctrl_c(process.pid)
                # os.kill(process.pid, signal.CTRL_BREAK_EVENT)  # 发送 CTRL_BREAK_EVENT 信号
            else:
                # Linux/Mac：发送 SIGTERM
                process.terminate()
                
            # 等待进程完全退出，确保文件写入完毕
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # 如果超时仍未退出，则强制结束
            process.kill()
            