# Ant_Bee_Detection
**二分类问题：实现对蚂蚁和蜜蜂的识别**

20250329

***

## 通过本地操作让GitHub同步删除已移除的文件

1. 首先，在本地工作目录中确认该文件已经被物理删除或者不再需要保留。
2. 执行以下命令将其标记为被删除状态并提交更改到版本库。

```
Bash
git rm read_data.py.bak
```

此命令会从索引中移除指定文件，并准备在下次提交时反映这一变化。

3. 提交此次变更至本地存储库，附带描述信息以便后续追踪记录原因。

```
Bash
git commit -m "Remove unused backup file read_data.py.bak"
```

上述命令将当前阶段的所有改动打包成一个新的提交对象存入历史日志里。

4. 将最新的修改推送到远端服务器上覆盖原有分支内容。

```
Bash
git push origin main
```

这里假设默认主干名称为 `main`；如果项目采用的是其他命名约定，则需替换相应参数名。

***

## **如何将本地文件夹的更新同步到 GitHub**

为了将本地文件夹中的更改同步到 GitHub 存储库，可以按照以下方法操作：

#### 初始化 Git 仓库

如果尚未初始化本地项目的 Git 仓库，则需要运行以下命令来完成初始化：

```
Bash
git init
```

此命令会在当前目录下创建一个新的 `.git` 文件夹，用于跟踪版本控制。

#### 添加远程存储库地址

假设已经有一个现有的 GitHub 项目，可以通过以下命令将其设置为远程存储库：

```
Bash
git remote add origin https://github.com/<username>/<repository>.git
```

其中 `<username>` 是您的 GitHub 用户名，而 `<repository>` 是目标存储库名称。如果服务器能够连接到 GitHub，则可以直接执行上述命令；否则可能需要手动下载并上传必要的配置文件至服务器环境。

#### 跟踪变更与提交

当您对本地文件进行了修改之后（比如编辑 `README.md` 或新增了一个名为 `new.txt` 的文档），这些变化不会自动被推送到远端分支上。因此，必须先通过如下步骤处理它们:

- **查看状态**: 查看哪些文件发生了改变以及处于何种阶段。

  ```
  Bash
  git status
  ```

- **暂存更改**: 将所有已修改或者新建好的文件加入索引区域(即准备提交的状态卡)。

  ```
  Bash
  git add .
  ```

- **提交消息**: 提交所作的所有更动，并附带一条描述性的日志记录下来。

  ```
  Bash
  git commit -m "Add new changes including README update and additional file"
  ```

#### 推送数据至上游分支

最后一步就是把刚才所做的工作成果推送回云端上的主干线上去吧！如果是第一次推送的话，记得指定好对应的branch name哦～比如说master或者是main之类的，默认情况下应该是后者啦～

```
Bash
git push -u origin main
```

这样就完成了整个流程：从初始建立链接到最后成功分享自己的劳动结晶给全世界人民欣赏的过程咯！

## 代码语法说明
1. `os.path.join` 的作用是自动处理路径分隔符，因为不同操作系统的路径分隔符不同：Windows为反斜杠`\`，Linux/macOS为正斜杠`/`，提高代码跨平台兼容性。
2. `def`定义下，字符串和代码块均需缩进
3. 一些操作技巧 ：
    - 按住`Ctrl`，单击方法，可得到说明文件；
    - `Ctrl`+`/` 注释该行；
    - `Shift`+`Enter`，换行；
4. 查看TensorBoard生成的图片：在Terminal中输入`tensorboard --logdir=logs --port=6007 or xxxx`