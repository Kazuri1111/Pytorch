
# -*- coding: utf-8 -*-

# .ipynbをimportする

# 使用例1:
# import ipynb_import_lib
# ipynb_import_lib.import_ipynb("./sample_lib.ipynb", "lib1")
# lib1.any_func1(...) # 呼び出し等

# 使用例2:
# import ipynb_import_lib
# lib1 = ipynb_import_lib.import_ipynb("./sample_lib.ipynb")
# lib1.any_func1(...) # 呼び出し等

import json, ast, re
import pathlib, sys

def main(): # 実行切替用
    pass

# sec: .ipynbをimport

# .ipynbファイルのpyコードをimport (import用の.py生成)
def import_ipynb(path_nb, # .ipynbのパス
    name=None, no_expr=True, # .pyファイル名, 計算式は除外
    if_import=True, must_import=False): # importしてglobalsに登録, 既存でも再import
    
    # sec: パス解決
    
    if not isinstance(path_nb, pathlib.Path):
        path_nb = pathlib.Path(path_nb)
    
    if name is None: # if: .ipynbと同名
        name = path_nb.stem
    
    path_py = path_nb.parent.joinpath("__pycache__", name + ".py")
    
    # sec: .ipynbファイル内容の更新有無
    
    if not path_py.exists() or \
        path_nb.stat().st_mtime > path_py.stat().st_mtime: # if: 更新あり
        
        # sec: .py生成
        
        text_code = get_code_from_ipynb(path_nb, no_expr)
        
        path_py.parent.mkdir(exist_ok=True) # __pycache__フォルダ作成
        
        with open(path_py, 'w', encoding='UTF8') as file:
            file.write(text_code)
    
    # sec: import
    
    if not if_import: # if: .py生成のみ
        return
    
    pygl = globals() # global域の変数
    if not must_import and name in pygl: # if: 既存 既にimport済み
        return pygl[name]
    
    else:
        sys.path.append(str(path_py.parent)) # __pycache__フォルダへのパス tag:CACH

        imported = __import__(name) # 注意: "abc.def"など下位を読み込めない
        # imported = importlib.import_module(name) # __import__で代用可 importlib要らず
        
        sys.path.remove(str(path_py.parent)) # 解除 tag:CACH

        pygl[name] = imported
        return imported

def test__import_ipynb():
    # 結果:
    # <module 'tut1' from '__pycache__\\tut1.py'>
    # <module 'tut1' from '__pycache__\\tut1.py'>
    # <class 'tut1.DQN'>
    # <module 'official-tut reinforcement_q_learning' from '__pycache__\\official-tut reinforcement_q_learning.py'>
    # <class 'official-tut reinforcement_q_learning.DQN'>

    PATH_NB = "./official-tut reinforcement_q_learning.ipynb"
    import_ipynb(PATH_NB, "tut1")
    print(tut1)
    import_ipynb(PATH_NB, "tut1") # 2回目
    print(tut1)
    print(tut1.DQN)

    tut2 = import_ipynb(PATH_NB)
    print(tut2)
    print(tut2.DQN)
# main = test__import_ipynb

# .ipynbファイルからpyコード(文字列)を取得
def get_code_from_ipynb(path_nb, # .ipynbのパス
    no_expr=True): # グローバル域に定義された計算式は除外
    # 非対応: """～多行～"""の文字列はコメントアウト不可
    
    # sec: load
    
    with open(path_nb, 'rb') as file:
        json_root = json.load(file)
    
    # sec: code cell
    
    re_ipy = re.compile(r"(^%|^!)", re.MULTILINE)
    text_code = ""
    for elem_i in json_root["cells"]:

        if elem_i["cell_type"] != "code":
            continue

        text = "".join(elem_i["source"])
        text = re_ipy.sub("# NOT-PY: \\1", text)
        
        text_code += "\n# ---------- cell ----------\n\n" + text + "\n"
        
    # print(text_code)
    
    if not no_expr: # if: コードそのまま
        return text_code
    
    # sec: AST
    
    text_code += "\npass" # HACK: 最終行の検出用に空要素を追加

    tree_root = ast.parse(text_code)

    # sec: 計算式
    
    i_curr = None
    node_curr = None
    expr_list = [] # グローバル域に定義された計算式範囲 [(i_from, i_to), ...]
    for node_i in ast.iter_child_nodes(tree_root):
        
        i_next = get_lineno(node_i) # 1つ前の要素の行番号範囲を決める
        if i_curr is not None: # if: 初回以外
            
            if node_curr.__class__.__name__ not in (
                "Import, ImportFrom, FunctionDef, ClassDef"):
            
                expr_list.append((i_curr, i_next - 1))
        
        i_curr, node_curr = i_next, node_i
    
    # HACK: 追加したpassは既に除去済み 1つ前の要素の為
    
    # print(expr_list)
    
    # sec: コードを行で分割
    
    re_line = re.compile(r"\r?\n")
    lines = re_line.split(text_code)
    lines = lines[:-1] # HACK: 追加したpassを除去
    
    # sec: 計算式をコメントアウト
    
    re_noco = re.compile(r"^(\s\t)*$|^(\s\t)*#")
    for i_from, i_to in expr_list:
        
        for i in range(i_from, i_to + 1):
            if not re_noco.search(lines[i]): # if: コードあり
                lines[i] = "# EXPR: " + lines[i]
    
    return "\n".join(lines)

def test__get_code_from_ipynb():
    # 結果:
    # 
    # # ---------- cell ----------
    # 
    # # NOT-PY: %matplotlib inline
    # 
    # # ---------- cell ----------
    # 
    # # NOT-PY: !pip show pip
    # 
    # # ---------- cell ----------
    # 
    # # import gym
    # import math
    # ...(以降略)
    
    PATH_NB = "./official-tut reinforcement_q_learning.ipynb"
    text_code = get_code_from_ipynb(PATH_NB)
    print(text_code)
# main = test__get_code_from_ipynb

# AST Nodeから行番号を取得
def get_lineno(node):
    
    try:
        return node.lineno - 1 # 開始番号は1からの為
    except:
        return None

# sec: entry

if __name__ == "__main__": main()

