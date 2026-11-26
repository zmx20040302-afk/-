# PMUT 阵列编程台

[打开在线工作台](https://zmx20040302-afk.github.io/-/#inverse)

在 50×50 固定阵列上计算 23 个频点的声场，从目标声场反求通电排布。逆向设计可指定随机分散、紧凑成团、边缘优先、条纹或不规则多斑块，也可选择全部五类。正式工作台采用保存的开路模型；理想短路边界用于另行复算。

## 在线使用

在浏览器中打开上方链接即可。首次打开会下载计算引擎、NumPy 和正向模型，首次逆向设计再加载排布模型。网页在当前设备上计算。选择全部五类后，浏览器依次显示各类结果。计算时间取决于设备性能，仓库内的历史基准使用 Windows 原生工作台测得。

## Windows 本机使用

1. 安装 64 位 Python（推荐 3.12），下载并解压整个仓库。
2. 双击 `Open_PMUT_Full_Field.cmd`。首次启动会创建项目独立环境并安装固定版本的 NumPy。以后使用本机版可离线计算。
3. 如需登录后自动启动，在项目目录运行 `Open_PMUT_Full_Field.cmd Install`。把 [本机工作台](http://127.0.0.1:8797/#inverse) 加入书签即可直接访问。运行 `Open_PMUT_Full_Field.cmd Uninstall` 可以取消自启。

本机版使用独立进程并行生成五类结果。更改仓库目录后，请在新位置重新执行安装命令。

[排布分类的数学定义](short_circuit/inverse_design/S308/classification.html) · [同批 15 个目标的数值验证报告](short_circuit/inverse_design/S321/report.html)
