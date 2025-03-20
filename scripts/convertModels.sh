#tools/converter.py --mode onnx -f configs/damoyolo_tinynasL18_Ns.py -c outputModels/damoyolo_tinynasL18_Ns_03182025.pth --batch_size 1
#tools/converter.py --mode onnx -f configs/damoyolo_tinynasL18_Nm.py -c outputModels/damoyolo_tinynasL18_Nm_03182025.pth --batch_size 1
#tools/converter.py --mode onnx -f configs/damoyolo_tinynasL20_Nl.py -c outputModels/damoyolo_tinynasL20_Nl_03182025.pth --batch_size 1
tools/converter.py --mode onnx -f configs/damoyolo_tinynasL20_T.py -c outputModels/models_03202025_320x320/damoyolo_tinynasL20_T.pth --batch_size 1
tools/converter.py --mode onnx -f configs/damoyolo_tinynasL25_S.py -c outputModels/models_03202025_320x320/damoyolo_tinynasL25_S.pth --batch_size 1
tools/converter.py --mode onnx -f configs/damoyolo_tinynasL35_M.py -c outputModels/models_03202025_320x320/damoyolo_tinynasL35_M.pth --batch_size 1
tools/converter.py --mode onnx -f configs/damoyolo_tinynasL45_L.py -c outputModels/models_03202025_320x320/damoyolo_tinynasL45_L.pth --batch_size 1
