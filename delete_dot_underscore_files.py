import os
import argparse

def delete_dot_underscore_files(target_dir):
    deleted_files = []
    for root, dirs, files in os.walk(target_dir):
        for filename in files:
            if filename.startswith("._"):
                file_path = os.path.join(root, filename)
                try:
                    os.remove(file_path)
                    deleted_files.append(file_path)
                    print(f"已删除: {file_path}")
                except Exception as e:
                    print(f"删除失败: {file_path}, 错误: {e}")
    print(f"共删除{len(deleted_files)}个文件。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="递归删除目录下所有以._开头的文件")
    parser.add_argument("dir", nargs="?", default=".", help="目标目录，默认当前目录")
    args = parser.parse_args()
    delete_dot_underscore_files(args.dir)
