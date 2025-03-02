import subprocess
import sys
import os

def main():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(current_dir, 'src/app.py')
        
        if not os.path.exists(app_path):
            print("错误：找不到app.py文件")
            sys.exit(1)
            
        subprocess.run([sys.executable, app_path], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"运行app.py时发生错误：{e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n程序已被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"发生未知错误：{e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 