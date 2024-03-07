import os
def get_last_filename_in_directory(directory):
    # 取得目錄中的所有檔案名稱
    files = os.listdir(directory)

    # 過濾掉非檔案的項目（可能包含子目錄等）
    files = [file for file in files if os.path.isfile(os.path.join(directory, file))]

    # 排序檔案名稱
    files.sort()

    if len(files) > 0:
        # 回傳最後一個檔案的名稱
        return files[-1]
    else:
        # 若目錄中沒有檔案，回傳None
        return None