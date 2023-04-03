import dgl
import numpy as np
import torch as th
import os

class DataPre:
    def __init__(self, path, mode = 'PNAConv', env_mode = 'with_BF', numoffacetype = 5):
        self.mode = mode
        self.env_mode = env_mode
        self.path = path
        self.numoffacetype = numoffacetype
        self.FaceTag = []

        self.StandardizeDict = dict()
        self.StandardizeInfoList = dict()
        self.StandardizeFaceTypes = []
        self.StandardizeFaceConvexities = []
        self.StandardizeFaceCenterCoors = []
        self.StandardizeFaceCenterVectors = []
        self.StandardizeRevolutionAxisVectors = []
        self.StandardizeFaceCompletes = []
        self.StandardizeFaceAreas = []
        self.StandardizeBaseFaces = []

        self.AllCoordinate = []
        self.AllCoordinateX = []
        self.AllCoordinateY = []
        self.AllCoordinateZ = []
        self.AllFaceArea = []

        self.PartInfoDict = dict()
        self.PartInfoDict["Nodes"] = []
        self.PartInfoDict["Edges"] = []
        self.PartInfoDict["EdgesOneDirection"] = []

        self.ulists = []
        self.vlists = []

        self.base_face_ids = []

        with open(self.path) as p:
            lines = p.readlines()
            for i in range(len(lines)):
                data = list(map(str, (lines[i].strip().split(' '))))
                NodeAttr = []
                EdgeAttr1 = []
                EdgeAttr2 = []
                if data[0] == "#N":
                    if int(data[1]) >= self.numoffacetype:
                        face_type = self.numoffacetype
                    else:
                        face_type = int(data[1])
                    if data[2] == "-1":
                        data[2] = 4
                    NodeAttr.append(face_type)
                    NodeAttr.append(int(data[2]))
                    NodeAttr.append(float(data[3]))
                    NodeAttr.append(float(data[4]))
                    NodeAttr.append(float(data[5]))
                    NodeAttr.append(float(data[6]))
                    NodeAttr.append(float(data[7]))
                    NodeAttr.append(float(data[8]))
                    NodeAttr.append(float(data[9]))
                    NodeAttr.append(float(data[10]))
                    NodeAttr.append(float(data[11]))

                    NodeAttr.append(int(data[12])) #Wether Complete

                    NodeAttr.append(float(data[13])) #Face Area

                    NodeAttr.append(int(data[14])) #Base Face

                    #NodeAttr.append(int(data[15])) #Tag
                    NodeAttr.append(data[15])  # Tag_Handle

                    if int(data[14]) == 1:
                        self.base_face_ids.append(i)

                    self.AllCoordinate.append(float(data[3]))
                    self.AllCoordinate.append(float(data[4]))
                    self.AllCoordinate.append(float(data[5]))
                    self.AllCoordinateX.append(float(data[3]))
                    self.AllCoordinateY.append(float(data[4]))
                    self.AllCoordinateZ.append(float(data[5]))
                    self.AllFaceArea.append(float(data[13]))

                    self.FaceTag.append(data[15])

                    self.PartInfoDict ["Nodes"].append(NodeAttr)
                elif data[0] == "#E":
                    if data[3] == "31":
                        data[3] = 3 # concave
                    elif data[3] == "32":
                        data[3] = 4 # convex
                    EdgeAttr1.append(int(data[1]))
                    EdgeAttr1.append(int(data[2]))
                    EdgeAttr1.append(int(data[3]))

                    EdgeAttr2.append(int(data[2]))
                    EdgeAttr2.append(int(data[1]))
                    EdgeAttr2.append(int(data[3]))

                    self.ulists.append(int(data[1]))
                    self.vlists.append(int(data[2]))

                    self.ulists.append(int(data[2]))
                    self.vlists.append(int(data[1]))

                    self.PartInfoDict["Edges"].append(EdgeAttr1)
                    self.PartInfoDict["Edges"].append(EdgeAttr2)

                    self.PartInfoDict["EdgesOneDirection"].append(EdgeAttr1)

    def one_hot(self, num, fullsize):
        One_Hot_Num = []
        for i in range(fullsize):
            if i != num-1:
                One_Hot_Num.append(0)
            else:
                One_Hot_Num.append(1)
        return One_Hot_Num

    def MinMaxNum(self):
        MinNum = min(self.AllCoordinate)
        MaxNum = max(self.AllCoordinate)
        return MinNum, MaxNum

    def MinMaxNumXYZ(self):
        MinNumX = min(self.AllCoordinateX)
        MaxNumX = max(self.AllCoordinateX)
        MinNumY = min(self.AllCoordinateY)
        MaxNumY = max(self.AllCoordinateY)
        MinNumZ = min(self.AllCoordinateZ)
        MaxNumZ = max(self.AllCoordinateZ)
        return MinNumX, MaxNumX, MinNumY, MaxNumY, MinNumZ, MaxNumZ

    def MinMaxArea(self):
        MinArea = min(self.AllFaceArea)
        MaxArea = max(self.AllFaceArea)
        return MinArea, MaxArea

    # 标准化
    def Standardize(self):
        TempNodes = self.PartInfoDict["Nodes"]
        MinCoorX, MaxCoorX, MinCoorY, MaxCoorY, MinCoorZ, MaxCoorZ = self.MinMaxNumXYZ()
        MinArea, MaxArea = self.MinMaxArea()
        NodeDictsList = []
        NodesList = []

        self.snbf_areas = dict()

        for i in range (len(TempNodes)):
            NodeDict = dict()
            NodeList = []
            StandardizeFaceType = []
            StandardizeFaceConvexity = []
            StandardizeFaceCenterCoor = []
            StandardizeFaceCenterVector = []
            StandardizeRevolutionAxisVector = []
            StandardizeFaceComplete = []
            StandardizeFaceArea = []
            StandardizeBaseFace = []

            TempNode = TempNodes[i]

            NodeDict["FaceType"] = self.one_hot(TempNode[0]-1, self.numoffacetype)
            NodeDict["FaceConvexity"] = self.one_hot(TempNode[1]-1, 4)
            if abs(MaxCoorX - MinCoorX) == 0:
                NomalX = 1.0
            else:
                NomalX = abs((TempNode[2]-MinCoorX)/abs(MaxCoorX - MinCoorX))

            if abs(MaxCoorY - MinCoorY) == 0:
                NomalY = 1.0
            else:
                NomalY = abs((TempNode[3]-MinCoorY)/abs(MaxCoorY - MinCoorY))

            if abs(MaxCoorZ - MinCoorZ) == 0:
                NomalZ = 1.0
            else:
                NomalZ = abs((TempNode[4]-MinCoorZ)/abs(MaxCoorZ - MinCoorZ))

            if abs(MaxArea-MinArea) == 0:
                NormalArea = 1.0
            else:
                NormalArea = abs((TempNode[12]-MinArea)/abs(MaxArea-MinArea))

            NodeDict["CenterPointCoordinate"] = [NomalX,NomalY,NomalZ]
            NodeDict["CenterPointVector"] = [TempNode[5], TempNode[6], TempNode[7]]
            NodeDict["RevolutionAxis"] = [TempNode[8], TempNode[9], TempNode[10]]
            NodeDict["WetherComplete"] = self.one_hot(TempNode[11]+1, 3)
            NodeDict["FaceArea"] = [NormalArea]
            NodeDict["BaseFace"] = [TempNode[13]]

            if TempNode[13] == 1:
                sna = [TempNode[5], TempNode[6], TempNode[7]]
                fa = abs((TempNode[12] - MinArea) / abs(MaxArea - MinArea))
                if str(sna) not in self.snbf_areas.keys():
                    self.snbf_areas[str(sna)] = fa
                else:
                    self.snbf_areas[str(sna)] += fa

            bf_areas = []
            for area in self.snbf_areas.values():
                bf_areas.append(area)
            self.max_bfs_normal = [0, 0, 0]
            if len(bf_areas) != 0:
                max_bfs_area = max(bf_areas)
                for k, v in self.snbf_areas.items():
                    if v == max_bfs_area:
                        self.max_bfs_normal = list(map(float,k.strip("[").strip("]").split(",")))
                        break

            NodeDictsList.append(NodeDict)

            for j in range(len(NodeDict["FaceType"])):
                NodeList.append(NodeDict["FaceType"][j])
                StandardizeFaceType.append(NodeDict["FaceType"][j])
            for j in range(len(NodeDict["FaceConvexity"])):
                NodeList.append(NodeDict["FaceConvexity"][j])
                StandardizeFaceConvexity.append(NodeDict["FaceConvexity"][j])
            for j in range(len(NodeDict["CenterPointCoordinate"])):
                NodeList.append(NodeDict["CenterPointCoordinate"][j])
                StandardizeFaceCenterCoor.append(NodeDict["CenterPointCoordinate"][j])
            for j in range(len(NodeDict["CenterPointVector"])):
                NodeList.append(NodeDict["CenterPointVector"][j])
                StandardizeFaceCenterVector.append(NodeDict["CenterPointVector"][j])
            for j in range(len(NodeDict["RevolutionAxis"])):
                NodeList.append(NodeDict["RevolutionAxis"][j])
                StandardizeRevolutionAxisVector.append(NodeDict["RevolutionAxis"][j])
            for j in range(len(NodeDict["WetherComplete"])):
                NodeList.append(NodeDict["WetherComplete"][j])
                StandardizeFaceComplete.append(NodeDict["WetherComplete"][j])
            for j in range(len(NodeDict["FaceArea"])):
                NodeList.append(NodeDict["FaceArea"][j])
                StandardizeFaceArea.append(NodeDict["FaceArea"][j])
            for j in range(len(NodeDict["BaseFace"])):
                NodeList.append(NodeDict["BaseFace"][j])
                StandardizeBaseFace.append(NodeDict["BaseFace"][j])

            NodesList.append(NodeList)
            self.StandardizeFaceTypes.append(StandardizeFaceType)
            self.StandardizeFaceConvexities.append(StandardizeFaceConvexity)
            self.StandardizeFaceCenterCoors.append(StandardizeFaceCenterCoor)
            self.StandardizeFaceCenterVectors.append(StandardizeFaceCenterVector)
            self.StandardizeRevolutionAxisVectors.append(StandardizeRevolutionAxisVector)
            self.StandardizeFaceCompletes.append(StandardizeFaceComplete)
            self.StandardizeFaceAreas.append(StandardizeFaceArea)
            self.StandardizeBaseFaces.append(StandardizeBaseFace)

        self.StandardizeDict["Nodes"] = NodeDictsList
        self.StandardizeInfoList["Nodes"] = NodesList

        EdgeDictsList = []
        EdgesList = []
        TempEdges = self.PartInfoDict["Edges"]
        for i in range (len(TempEdges)):
            EdgeDict = dict()
            TempEdge = TempEdges[i]
            if (TempEdge[2] == 1)|(TempEdge[2] == 3):
                TempEdge[2] = 1 # Concave [1, 0]
            else:
                TempEdge[2] = 2 # Convex [0, 1]

            EdgeDict["NodesID"] = [TempEdge[0], TempEdge[1]]
            if (self.mode == 'PNAConv')|(self.mode == 'EGATConv'):
                EdgeDict["EdgeConvexity"] = self.one_hot(TempEdge[2], 2)
            elif self.mode == 'GINEConv':
                EdgeDict["EdgeConvexity"] = -TempEdge[2]+2 # Concave:1, Convex:0

            EdgeDictsList.append(EdgeDict)
            EdgesList.append(EdgeDict["EdgeConvexity"])

        self.StandardizeDict["Edges"] = EdgeDictsList
        self.StandardizeInfoList["Edges"] = EdgesList

    def Dict2Graph_Union(self):
        u, v = th.tensor(self.ulists), th.tensor(self.vlists)
        g = dgl.graph((u, v))
        if self.env_mode == 'with_BF':
            zeros = np.zeros((len(th.tensor(self.StandardizeInfoList["Nodes"])), 2))
            g.ndata["NodeAttr"] = th.cat((th.tensor(self.StandardizeInfoList["Nodes"], dtype=th.float32),th.tensor(zeros)), dim=1)
        elif self.env_mode == 'without_BF':
            zeros = np.zeros((len(th.tensor(self.StandardizeInfoList["Nodes"])), 1))
            g.ndata["NodeAttr"] = th.cat((th.tensor(self.StandardizeInfoList["Nodes"], dtype=th.float32), th.tensor(zeros)), dim=1)

        g.edata["EdgeAttr"] = th.tensor(self.StandardizeInfoList["Edges"], dtype=th.float32)
        return g, self.snbf_areas

    def Dict2Graph_Union_v2(self):
        u, v = th.tensor(self.ulists), th.tensor(self.vlists)
        g = dgl.graph((u, v))
        if self.env_mode == 'with_BF':
            zeros = np.zeros((len(th.tensor(self.StandardizeInfoList["Nodes"])), 2))
            g.ndata["NodeAttr"] = th.cat((th.tensor(self.StandardizeInfoList["Nodes"], dtype=th.float32),th.tensor(zeros)), dim=1)
        elif self.env_mode == 'without_BF':
            zeros = np.zeros((len(th.tensor(self.StandardizeInfoList["Nodes"])), 1))
            g.ndata["NodeAttr"] = th.cat((th.tensor(self.StandardizeInfoList["Nodes"], dtype=th.float32), th.tensor(zeros)), dim=1)

        g.edata["EdgeAttr"] = th.tensor(self.StandardizeInfoList["Edges"], dtype=th.float32)
        return g, self.max_bfs_normal, self.base_face_ids

    def Dict2Graph_Split(self):
        u, v = th.tensor(self.ulists), th.tensor(self.vlists)
        g = dgl.graph((u, v))
        g.ndata["FaceType"] = th.tensor(self.StandardizeFaceTypes)
        g.ndata["FaceConvexity"] = th.tensor(self.StandardizeFaceConvexities)
        g.ndata["FaceCenterCoordinate"] = th.tensor(self.StandardizeFaceCenterCoors)
        g.ndata["FaceCenterVector"] = th.tensor(self.StandardizeFaceCenterVectors)
        g.ndata["RevolutionAxis"] = th.tensor(self.StandardizeRevolutionAxisVectors)
        g.ndata["WetherComplete"] = th.tensor(self.StandardizeFaceCompletes)
        g.ndata["FaceArea"] = th.tensor(self.StandardizeFaceAreas)
        g.ndata["BaseFace"] = th.tensor(self.StandardizeBaseFaces)

        g.edata["EdgeAttr"] = th.tensor(self.StandardizeInfoList["Edges"])
        return g, self.snbf_areas

    def get_snbf_areas(self):
        return self.snbf_areas


def ReadDataFromPath(FilesPath, Mode, EnvMode):
    TempPath = FilesPath
    files = os.listdir(TempPath)

    Graphs = []
    snbf_areas = []
    for file in files:
        DP = DataPre(TempPath + '\\' + file, Mode, EnvMode)
        DP.Standardize()
        Graph, snbf_area = DP.Dict2Graph_Union()
        Graphs.append(Graph)
        snbf_areas.append(snbf_area)
        del(DP)
    return Graphs, snbf_areas

def ReadDataFromPath_v2(FilesPath, Mode, EnvMode):
    TempPath = FilesPath
    files = os.listdir(TempPath)

    Graphs = []
    max_bfs_normals = []
    base_face_ids = []
    for file in files:
        DP = DataPre(TempPath + '\\' + file, Mode, EnvMode)
        DP.Standardize()
        Graph, max_bfs_normal, base_face_id = DP.Dict2Graph_Union_v2()
        Graphs.append(Graph)
        max_bfs_normals.append(max_bfs_normal)
        base_face_ids.append(base_face_id)
        del(DP)
    return Graphs, max_bfs_normals, base_face_ids


class Tool_Direction:
    def __init__(self, face_normals, SpherePoints, device):
        self.device = device
        self.face_normals = th.tensor(face_normals).to(device)
        self.sphere_points = SpherePoints.to(device)


    def judge_tool_cover(self, tolerance = 0.05):
        judge_matrix = self.face_normals.mm(self.sphere_points)

        min_column = th.min(judge_matrix, dim = 0)[0]
        mask_min_column = min_column > -tolerance
        mask_min_column =  th.tensor(mask_min_column, dtype=int)
        sum_mask = sum(mask_min_column)
        if sum_mask == 0:
            return 0
        else:
            return 1

class Tool_Direction_v1:
    def __init__(self, face_normals, base_face_normal):
        self.face_normals = np.array(face_normals)
        self.base_face_normal = np.array(base_face_normal)

    def judge_tool_cover(self):
        tbfn = self.base_face_normal.transpose()
        judge_vector = np.dot(self.face_normals, tbfn)
        mask_judge_vector = judge_vector < 0.0
        mask_judge_vector = np.array(mask_judge_vector, dtype=int)
        sum_mask = sum(mask_judge_vector)
        if sum_mask == 0:
            return 1
        else:
            return 0
