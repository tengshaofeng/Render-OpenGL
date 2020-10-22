# coding=utf-8
import sys, pygame
from pygame.locals import *
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
from math import *
import cv2, io
from PIL import Image


class Render(object):
    def __init__(self):
        self.width = None
        self.height = None
        self.m_dynamic_tex_id = None
        self.m_fbo = None
        self.m_depth_rb = None
        self.m_mesh_vert_id = None
        self.m_mesh_uv_id = None
        self.m_mesh_faces_id = None
        self.m_mesh_tex_id = None
        self.m_mesh_normals_id = 0
    def render_init(self, width=224, height=224):
        ######1 render init
        # #clearFrameBuffer()#
        self.width = width
        self.height = height

        # # Create a texture
        
        if self.m_dynamic_tex_id is None:
            self.m_dynamic_tex_id = glGenTextures(1)# # 我需要1个纹理对象，self.m_dynamic_tex_id存纹理对象的索引 tbq
        glBindTexture(GL_TEXTURE_2D, self.m_dynamic_tex_id)##告诉opengl下面代码中对2d纹理的操作都是针对该索引对象 tbq
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)##s方向上的贴图模式，忽略边框纹理 tbq
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)##T方向上的贴图模式，忽略边框纹理 tbq
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)##纹理过滤， 放大过滤 tbq
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)##纹理过滤，缩小过滤 tbq


        # Allocate texture storage
        glBindTexture(GL_TEXTURE_2D, self.m_dynamic_tex_id)#
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, None)#

        # Create framebuffer

        if self.m_fbo is None:
            self.m_fbo = glGenFramebuffers(1)#
        glBindFramebuffer(GL_FRAMEBUFFER, self.m_fbo)#

        # Attach 2D texture to this FBO
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.m_dynamic_tex_id, 0)#

        # Create renderbuffer
        if self.m_depth_rb is None:
            self.m_depth_rb = glGenRenderbuffers(1) #
        glBindRenderbuffer(GL_RENDERBUFFER, self.m_depth_rb)#
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)#

        # Attach depth buffer to FBO
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.m_depth_rb)#

        # Clean up
        glBindFramebuffer(GL_FRAMEBUFFER, 0)#

        # Setup viewport
        glViewport(0, 0, width, height)#

        # Settings
        glEnable(GL_DEPTH_TEST)#
        glEnable(GL_TEXTURE_2D)#

        # Enable blending
        glEnable(GL_BLEND)#
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)# # 在alpha空间融合

    def set_projection(self):
        f = (self.width+self.height)*2.5/5 #2560
        z_near = 10
        z_far = 2000000.0
        glMatrixMode(GL_PROJECTION)# # 接下来中矩阵操作中都在投影矩阵栈中进行 tbq
        glLoadIdentity()# # 用单位矩阵，对角线为1的矩阵代替当前矩阵 tbq

        #flip y, because image y increases downward, whereas camera y increases upward
        glScaled(1, -1, 1)#
        fovy = 2.0 * atan(self.height / (2.0 * f)) * 180.0 / pi #
        aspect = self.width / float(self.height)#
        gluPerspective(fovy, aspect, float(z_near), float(z_far))#

    def set_projection_tbq(self):
        f = 2560 # (self.width+self.height)*2.5/5 #2560
        z_near = 10.0
        z_far = 10000.0
        glMatrixMode(GL_PROJECTION)# # 接下来中矩阵操作中都在投影矩阵栈中进行 tbq
        glLoadIdentity()# # 用单位矩阵，对角线为1的矩阵代替当前矩阵 tbq

        #flip y, because image y increases downward, whereas camera y increases upward
        glScaled(1, -1, 1)#
        fovy = 2.0 * atan(self.height / (2.0 * f)) * 180.0 / pi #
        aspect = self.width / float(self.height)#
        gluPerspective(fovy, aspect, float(z_near), float(z_far))#

    def set_mesh(self):

        if self.m_mesh_vert_id is None:
            self.m_mesh_vert_id = glGenBuffers(1)
        self.m_mesh_total_vertices = len(vs)
        glBindBuffer(GL_ARRAY_BUFFER, self.m_mesh_vert_id)
        glBufferData(GL_ARRAY_BUFFER, 3*len(vs) * vs[0][0].nbytes,
                     vs, GL_STATIC_DRAW)

        # Initialize texture coordinates

        if self.m_mesh_uv_id is None:
            self.m_mesh_uv_id = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.m_mesh_uv_id)
            glBufferData(GL_ARRAY_BUFFER, 3*len(uv)* uv[0][0].nbytes,
                         uv, GL_STATIC_DRAW)

            # Initialize faces (triangles only)

        if self.m_mesh_faces_id is None:
            self.m_mesh_faces_id = glGenBuffers(1)
            self.m_mesh_total_faces = len(faces)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.m_mesh_faces_id)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(faces.flatten()) * faces.flatten()[0].nbytes, faces, GL_STATIC_DRAW) # unsigned short



        if self.m_mesh_tex_id is None:
            self.m_mesh_tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.m_mesh_tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        # Allocate texture storage or upload pixel data
        glBindTexture(GL_TEXTURE_2D, self.m_mesh_tex_id)


        self.m_mesh_tex_width = self.m_mesh_tex_height = 0
        if (self.m_mesh_tex_width != tex.shape[1] or self.m_mesh_tex_height != tex.shape[0]):
            self.m_mesh_tex_width = tex.shape[1]
            self.m_mesh_tex_height = tex.shape[0]
            if tex.shape[2] == 3:  # BGR
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.m_mesh_tex_width, self.m_mesh_tex_height,
                             0, GL_BGR, GL_UNSIGNED_BYTE, tex)


    def render(self, vec_t=[0, 0, -1030], vec_r = [55 / 100.0 * (pi/25), 0, 0]):
        """

        :param vec_t: 最后一个数设置渲染尺度，-1030/2图片比-1030尺度放大一倍
        :param vec_r:
        """
        print(vec_r, vec_t)
        glBindFramebuffer(GL_FRAMEBUFFER, self.m_fbo)#
        glClearColor(0.0, 0.0, 0.0, 0.0)#
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)#

        glMatrixMode(GL_MODELVIEW)#
        glPushMatrix()#

        #Translations
        #glTranslatef(vec_t[0], vec_t[1], vec_t[2])#
        glTranslatef(vec_t[0], vec_t[1],vec_t[2])#  tbq 其实是计算出来的 猜测：要小于vertices 的z轴最小值， -250000

        #Axis angle
        rx = vec_r[0]#
        ry = vec_r[1]#
        rz = vec_r[2]#
        angle = float(sqrt(rx * rx + ry * ry + rz * rz))#
        if(abs(angle) > 1e-6):
            glRotatef(angle / pi*180.0, rx / angle, ry / angle, rz / angle)#

        self.drawMesh()#

        glPopMatrix()#
        glFlush()#
        glBindFramebuffer(GL_FRAMEBUFFER, 0)##


    def get_frame_buffer(self):
        img = np.zeros((self.height, self.width, 4)).astype(np.uint8)
        glBindFramebuffer(GL_FRAMEBUFFER, self.m_fbo)
        x, y, width, height = glGetIntegerv(GL_VIEWPORT)  # tbq add
        glPixelStorei(GL_PACK_ALIGNMENT, 1)   # tbq add
        data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, None) # tbq add
        # data = glReadPixels(0, 0, self.width, self.height, GL_BGR, GL_UNSIGNED_BYTE, None) # tbq comment
        image = Image.frombytes("RGBA", (self.width, self.height), data)
        # image = image.transpose(Image.FLIP_TOP_BOTTOM)
        # image_data=io.BytesIO(img)
        # pil_img = Image.open(image_data).convert('RGB')
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        image.save('./result/output0.png')
        # cv2.imwrite('./result/output.jpg', img)
        return img



    def load_obj(self, fname='./result/shape(1).obj'):
        v_list = []
        vn_list = []
        face_list = []
        vt_list = []
        lines = open(fname).readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            if line.startswith('vn'):
                line_spt = line.rstrip().split(' ')[1:]
                line_spt = [float(t) for t in line_spt]
                vn_list.append(line_spt)
            elif line.startswith('vt'):
                line_spt = line.rstrip().split(' ')[1:]
                line_spt = [float(t) for t in line_spt]
                vt_list.append(line_spt)
            elif line.startswith('v'):
                line_spt = line.rstrip().split(' ')[1:]
                line_spt = [float(t) for t in line_spt]
                v_list.append(line_spt)
            elif line.startswith('f'):
                line_spt = line.rstrip().split(' ')[1:]
                line_spt = [int(t.split('//')[0]) for t in line_spt]
                face_list.append(line_spt)

        return np.array(v_list).astype(np.float32), np.array(vn_list).astype(np.float32), np.array(face_list).astype(np.int32), np.array(vt_list).astype(np.float32)

    def drawMesh(self):


        # glBegin(GL_TRIANGLES)
        # glColor4f(0, 1, 0, 1)
        # for facet in faces:
        #     for vIdx in facet:
        #         glVertex3d(vs[vIdx, 0], vs[vIdx, 1], 0)
        # glEnd()

        #glColor4f(0, 1, 0, 1)

        glEnableClientState(GL_VERTEX_ARRAY)#
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)#
        if (self.m_mesh_normals_id > 0):
            glEnableClientState(GL_NORMAL_ARRAY)#


        glBindTexture(GL_TEXTURE_2D, self.m_mesh_tex_id)#

        glBindBuffer(GL_ARRAY_BUFFER, self.m_mesh_vert_id)#
        glVertexPointer(3, GL_FLOAT, 0, None)#

        glBindBuffer(GL_ARRAY_BUFFER, self.m_mesh_uv_id)#
        glTexCoordPointer(2, GL_FLOAT, 0, None)#

        if (self.m_mesh_normals_id > 0):
            glBindBuffer(GL_ARRAY_BUFFER, self.m_mesh_normals_id)#
            glNormalPointer(GL_FLOAT, 0, None)#


        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.m_mesh_faces_id)#

        glDrawElements(GL_TRIANGLES, self.m_mesh_total_faces * 3, GL_UNSIGNED_SHORT, None)#

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)#//
        glBindBuffer(GL_ARRAY_BUFFER, 0)#//



        glDisableClientState(GL_VERTEX_ARRAY)#
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)#
        if (self.m_mesh_normals_id > 0):
            glDisableClientState(GL_NORMAL_ARRAY)#

render_obj = Render()
# vs, vns, faces = render_obj.load_obj()
# vs, vns, faces, uv = render_obj.load_obj('./result/output.obj')
# faces = faces.astype(np.uint16)
# tex = cv2.imread('./result/fbb_tex.jpg')


vs, vns, faces, uv = render_obj.load_obj('./result/fangbb/output.obj')
faces = faces.astype(np.uint16)
tex = cv2.imread('./result/fangbb/tex.jpg')
# faces -= 1
# vs = np.load('./result/vertices.npy').astype(np.float32)
# faces = np.load('./result/faces.npy').astype(np.uint16)
# uv = np.load('./result/uv.npy').astype(np.float32)/224.0
# tex = cv2.imread('./result/tex.jpg')
# #vs[:, -1] = 0


# for facet in faces:
#     swap = facet[0]
#     facet[0] = facet[2]
#     facet[2] = swap

# pygame.init()
viewport = (512,512)
# # hx = viewport[0]/2
# # hy = viewport[1]/2
srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)
# glutInit(sys.argv)
# glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
g_width = 512
# glutInitWindowSize(g_width,g_width)
# glutInitWindowPosition(100//2,100//2)
# glutCreateWindow("OpenGL Environment")
# glutHideWindow()



# glutCreateMenu()
# err = glewInit()
# #


# # 以下设置视点，修复没有耳朵的情况
# EYE = np.array([0.0, 0.0, 1])                     # 眼睛的位置（默认z轴的正方向）
# LOOK_AT = np.array([0.0, 0.0, 0.0])
# # LOOK_AT = np.array([0.8, 0.8, 0.0])                 # 瞄准方向的参考点（默认在坐标原点）
# EYE_UP = np.array([0.0, 1.0, 0.0])                  # 定义对观察者而言的上方（默认y轴的正方向）
# # 设置视点
# gluLookAt(
#     EYE[0], EYE[1], EYE[2],
#     LOOK_AT[0], LOOK_AT[1], LOOK_AT[2],
#     EYE_UP[0], EYE_UP[1], EYE_UP[2]
# )

######1 render init
render_obj.render_init(g_width, g_width)

######2 setProjection (float f, float z_near, float z_far) f=2560, near=1, far=10000
render_obj.set_projection_tbq()


#####3  set mesh
render_obj.set_mesh()



#####4 render  void FaceRenderer::render(const cv::Mat& vecR, const cv::Mat& vecT)
render_obj.render() # vec_t=[0, 0, -100]

##### 5 get frame buffer
render_obj.get_frame_buffer()