import pybullet as p
import time
import numpy as np
import matplotlib.pyplot as plt
from camera import Camera
import cv2
import cv

IMG_SIDE = 300
IMG_HALF = IMG_SIDE/2
MARKER_LENGTH = 0.1
MARKER_CORNERS_WORLD = np.array(
    [
        [-MARKER_LENGTH/2,0.0,MARKER_LENGTH/2,1],
        [MARKER_LENGTH/2,0.0,MARKER_LENGTH/2,1],
        [MARKER_LENGTH/2,0.0,-MARKER_LENGTH/2,1],
        [-MARKER_LENGTH/2,0.0,-MARKER_LENGTH/2,1]
    ]
)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

def computeInterMatrix(Y, sd0):
    L = np.zeros((8,3))
    for idx in range(4):
        x = sd0[2*idx, 0]
        z = sd0[2*idx+1, 0]
        L[2*idx] = np.array([-1/Y, 0, z])
        L[2*idx+1] = np.array([0, -1/Y, -x])
    return L

def updateCamPos(cam):
    linkState = p.getLinkState(boxId, linkIndex=6)
    # pos
    xyz = linkState[0]
    # orientation
    quat = linkState[1]
    rotMat = p.getMatrixFromQuaternion(quat)
    rotMat = np.reshape(np.array(rotMat),(3,3))
    camera.set_new_position(xyz, rotMat)

camera = Camera(imgSize = [IMG_SIDE, IMG_SIDE])

dt = 1/240 # pybullet simulation step
q0 = 0.5  # starting position (radian)

L, m, g = 0.5, 1, 10
Y0 = 0.3
maxTime = 4
logTime = np.arange(0.0, maxTime, dt)
jointIndices = [1,3,5]
eefLinkIdx = 6

# or p.DIRECT for non-graphical version
physicsClient = p.connect(p.GUI,
                          options="--background_color_red=1 --background_color_blue=1 --background_color_green=1")
p.resetDebugVisualizerCamera(
    cameraDistance=0.8,
    cameraYaw=180,
    cameraPitch=0,
    cameraTargetPosition=[0.5, 0.3, 1]
)
p.setGravity(gravX=0, gravY=0, gravZ=-10)
boxId = p.loadURDF(fileName="./simple.urdf", useFixedBase=True)
p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)
p.changeDynamics(boxId, 2, linearDamping=0, angularDamping=0)
p.changeDynamics(boxId, 3, linearDamping=0, angularDamping=0)
p.changeDynamics(boxId, 4, linearDamping=0, angularDamping=0)
p.changeDynamics(boxId, 5, linearDamping=0, angularDamping=0)

# add aruco cube and aruco texture
c = p.loadURDF('aruco.urdf', (0.5, -0.3, 1), useFixedBase=True)
x = p.loadTexture('aruco_cube.png')
p.changeVisualShape(c, -1, textureUniqueId=x)
p.stepSimulation()


#Валидация линеаризации
# go to the starting position
p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetPositions=[0.5, 0.3, 0],
                            controlMode=p.POSITION_CONTROL)
for _ in range(100):
    p.stepSimulation()
p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetVelocities=[0, 0, 0],
                            controlMode=p.VELOCITY_CONTROL, forces=[0, 0, 0])
veloc1, veloc2, veloc3 = [], [], []
prev_vel = 0
a1, a2 = [], []
for t in range(len(logTime)):
    jStates = p.getJointStates(boxId, jointIndices=jointIndices)
    jPos = [state[0] for state in jStates]
    q1, q2 = jPos[0], jPos[1]
    a1.append(q1)
    a2.append(jPos[2])
    jVel = [state[1] for state in jStates]
    vel1, vel2, vel3 = jVel[0], jVel[1], jVel[2]
    derivative = (vel3 - prev_vel)/dt
    veloc1.append(vel1)
    veloc2.append(vel2)
    veloc3.append(vel3)
    C = np.array([[- 2 * m * L ** 2 * np.sin(q2) * vel1 * vel2 - m * L ** 2 * np.sin(q2) * vel2 ** 2],
                  [m * L ** 2 * vel1 ** 2 * np.sin(q2)],
                  [0]])
    G = np.array([[2 * m * g * L * np.sin(q1) + m * g * L * np.sin(q1 + q2)],
                  [m * g * L * np.sin(q1 + q2)],
                  [0]])
    M = np.array([[3 * m * L ** 2 + 2 * np.cos(q2) * m * L ** 2, m * L ** 2 + np.cos(q2) * m * L ** 2, 0],
                               [m * L ** 2 + np.cos(q2) * m * L ** 2, m * L ** 2, 0],
                               [0, 0, 6.666666666666666e-07]])
    u = C + G + np.array([[vel1], [vel2], [vel3]]) + M @ (np.array([[0.1], [0.1], [0.1]]))
    p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, controlMode=p.TORQUE_CONTROL,
                                forces=[u[0, 0], u[1, 0], u[2, 0]])
    p.stepSimulation()
    prev_vel = vel3
    #time.sleep(dt)

print(derivative)
plt.plot(logTime, veloc1, label=r'$\dot{\theta_1}$', color='orange')
plt.plot(logTime, veloc2, label=r'$\dot{\theta_2}$', color='b')
plt.plot(logTime, veloc3, label=r'$\dot{\theta_3}$', color='k')
plt.xlabel('t, s')
plt.ylabel(r'$\dot\theta$, m/s')
plt.grid()
plt.legend()
plt.show()


#Валидация движения в декартову точку в скоростном управлении
p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetPositions=[1.5, 1, 1],
                            controlMode=p.POSITION_CONTROL)
# p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetPositions=[0, -np.pi/2, 0],
#                             controlMode=p.POSITION_CONTROL)
for _ in range(100):
    p.stepSimulation()
p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetVelocities=[0, 0, 0],
                            controlMode=p.VELOCITY_CONTROL, forces=[0, 0, 0])
des_x = 0.6
des_z = 0.8
dphi = 0.5
# des_x = 0.55
# des_z = 1
# dphi = 0.2
xarr, zarr, phiarr = [], [], []
for i in range(len(logTime)):
    states = p.getJointStates(bodyUniqueId=boxId, jointIndices=jointIndices)
    angle = [state[0] for state in states]
    q1, q2, phi = angle[0], angle[1], angle[2]
    phiarr.append(phi)
    linkState = p.getLinkState(boxId, linkIndex=4)
    xSim2 = linkState[0][0]
    xarr.append(xSim2)
    zSim2 = linkState[0][2]
    zarr.append(zSim2)
    J = np.array([[-L * np.cos(q1) - L * np.cos(q1 + q2), -L * np.cos(q1 + q2), 0],
                  [L * np.sin(q1) + L * np.sin(q1 + q2), L * np.sin(q1 + q2), 0],
                  [1, 1, 1]])
    w = 10 * np.linalg.inv(J) @ -np.array([[xSim2 - des_x], [zSim2 - des_z], [phi - dphi]])
    p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetVelocities=[w[0,0], w[1,0], w[2,0]], controlMode=p.VELOCITY_CONTROL)
    p.stepSimulation()
    #time.sleep(dt)
fig, axs = plt.subplots(3, 1)
print(q1, q2, phi)
axs[0].plot(logTime, xarr, 'k')
axs[0].axhline(y=des_x, color='r', linestyle='--')
axs[0].set_title('X')
axs[0].grid()
axs[0].set_xlabel('t, s')
axs[0].set_ylabel('x, m')

axs[1].plot(logTime, zarr, 'k')
axs[1].axhline(y=des_z, color='r', linestyle='--')
axs[1].set_title('Z')
axs[1].grid()
axs[1].set_xlabel('t, s')
axs[1].set_ylabel('z, m')

axs[2].plot(logTime, phiarr, 'k')
axs[2].axhline(y=dphi, color='r', linestyle='--')
axs[2].set_title(r'$\phi$ ')
axs[2].grid()
axs[2].set_xlabel('t, s')
axs[2].set_ylabel(r'$\phi$, rad')

plt.show()


#Визуальное позиционирование
#Первая точка
# xd, zd = 0.5, 1
# qd = 0

#Вторая точка
# xd, zd = 0.55, 1
# qd = 0.2

#Третья точка
xd, zd = 0.45, 1
qd = -0.1

# go to the desired position
#p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetPositions=[0.0, -np.pi/2, qd],controlMode=p.POSITION_CONTROL)
#p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetPositions=[-0.1, -1.465, 0.2],controlMode=p.POSITION_CONTROL)
p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetPositions=[0.1, -1.6665, -0.1], controlMode=p.POSITION_CONTROL)
for _ in range(1000):
    p.stepSimulation()
    #time.sleep(dt)

updateCamPos(camera)
img = camera.get_frame()
corners, markerIds, rejectedCandidates = detector.detectMarkers(img)
sd0 = np.reshape(np.array(corners[0][0]), (8, 1))
sd0 = np.array([(s - IMG_HALF) / IMG_HALF for s in sd0])
sd0_x = np.concatenate((sd0[::2], [sd0[0]]), axis=0)
sd0_z = np.concatenate((sd0[1::2], [sd0[1]]), axis=0)
sd = np.reshape(np.array(corners[0][0]), (8, 1)).astype(int)

# Постоянная матрица L
# go to the starting position

#p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetPositions=[-0.1, -np.pi/2 + 0.1, 0.1], controlMode=p.POSITION_CONTROL)
p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetPositions=[0, -np.pi/2, 0.1], controlMode=p.POSITION_CONTROL)
for _ in range(100):
    p.stepSimulation()
p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetVelocities=[0, 0, 0],
                            controlMode=p.VELOCITY_CONTROL, forces=[0, 0, 0])
idx = 1
camCount = 0
k = 300
L0 = computeInterMatrix(Y0, sd0)
L0T = np.linalg.inv(L0.T @ L0) @ L0.T

logs0_x, logs0_z, logPosx, logPosz, logPosf = [], [], [], [], []
w = np.zeros((3, 1))
for t in range(len(logTime)):
    p.stepSimulation()
    camCount += 1
    if (camCount == 5):
        camCount = 0
        updateCamPos(camera)
        camera.get_frame()
        img = camera.get_frame()
        corners, markerIds, rejectedCandidates = detector.detectMarkers(img)
        s = corners[0][0, 0]
        s0 = np.reshape(np.array(corners[0][0]), (8, 1))
        s0 = np.array([(ss - IMG_HALF) / IMG_HALF for ss in s0])
        s0_x = s0[::2]
        s0_z = s0[1::2]
        logs0_x.append(s0_x)
        logs0_z.append(s0_z)
        e = s0 - sd0
        coef = np.diag([1, 1, 1.5])
        w = -coef @ L0T @ e
    jStates = p.getJointStates(boxId, jointIndices=jointIndices)
    jPos = [state[0] for state in jStates]
    jVel = [state[1] for state in jStates]
    q1, q2 = jPos[0], jPos[1]
    vel1, vel2, vel3 = jVel[0], jVel[1], jVel[2]
    logPosf.append(jPos[2])
    logPosx.append(p.getLinkState(bodyUniqueId=boxId, linkIndex=6)[0][0])
    logPosz.append(p.getLinkState(bodyUniqueId=boxId, linkIndex=6)[0][2])
    J = np.array([[-L * np.cos(q1) - L * np.cos(q1 + q2), -L * np.cos(q1 + q2), 0],
                  [L * np.sin(q1) + L * np.sin(q1 + q2), L * np.sin(q1 + q2), 0],
                  [1, 1, 1]])

    dq = (np.linalg.inv(J) @ w).flatten()[[1, 0, 2]]
    dq[0] = -dq[0]
    dq[2] = -dq[2]

    C = np.array([[- 2 * m * L ** 2 * np.sin(q2) * vel1 * vel2 - m * L ** 2 * np.sin(q2) * vel2 ** 2],
                  [m * L ** 2 * vel1 ** 2 * np.sin(q2)],
                  [0]])
    G = np.array([[2 * m * g * L * np.sin(q1) + m * g * L * np.sin(q1 + q2)],
                  [m * g * L * np.sin(q1 + q2)],
                  [0]])
    M = np.array([[3 * m * L ** 2 + 2 * np.cos(q2) * m * L ** 2, m * L ** 2 + np.cos(q2) * m * L ** 2, 0],
                  [m * L ** 2 + np.cos(q2) * m * L ** 2, m * L ** 2, 0],
                  [0, 0, 6.666666666666666e-07]])
    u = C + G + np.array([[vel1], [vel2], [vel3]]) + M @ (k * -np.array([[vel1 - dq[0]], [vel2 - dq[1]], [vel3 - dq[2]]]))
    p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, controlMode=p.TORQUE_CONTROL,
                                forces=[u[0, 0], u[1, 0], u[2, 0]])
    #time.sleep(dt)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(logs0_x, logs0_z, s=2, label='trajecory')
plt.plot(sd0_x, sd0_z, 'r', linewidth=3, label='desired position')
logs0_x = np.concatenate((logs0_x[0], [logs0_x[0][0]]), axis=0)
logs0_z = np.concatenate((logs0_z[0], [logs0_z[0][0]]), axis=0)
plt.plot(logs0_x, logs0_z, linestyle='--', linewidth=3, label='starting position')
plt.legend(loc='upper right')
plt.xlabel('ImgX'), plt.ylabel('ImgZ')
plt.grid()

plt.subplot(3, 2, 2)
plt.plot(logTime, logPosx, 'k')
plt.axhline(y=xd, color='r', linestyle='--')
plt.title('X')
plt.grid()
plt.xlabel('t, s')
plt.ylabel('x, m')

plt.subplot(3, 2, 4)
plt.plot(logTime, logPosz, 'k')
plt.axhline(y=zd, color='r', linestyle='--')
plt.title('Z')
plt.grid()
plt.xlabel('t, s')
plt.ylabel('z, m')

plt.subplot(3, 2, 6)
plt.plot(logTime, logPosf, 'k')
plt.axhline(y=qd, color='r', linestyle='--')
plt.title(r'$\phi$ ')
plt.grid()
plt.xlabel('t, s')
plt.ylabel(r'$\phi$, rad')

plt.show()


# go to the starting position

#p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetPositions=[-0.1, -np.pi/2 + 0.1, 0.1], controlMode=p.POSITION_CONTROL)
p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetPositions=[0, -np.pi/2, 0.1], controlMode=p.POSITION_CONTROL)
for _ in range(100):
    p.stepSimulation()
p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetVelocities=[0, 0, 0],
                            controlMode=p.VELOCITY_CONTROL, forces=[0, 0, 0])
idx = 1
camCount = 0
k = 300
logs0_x, logs0_z, logPosx, logPosz, logPosf = [], [], [], [], []
w = np.zeros((3, 1))
for t in range(len(logTime)):
    p.stepSimulation()
    camCount += 1
    if (camCount == 5):
        camCount = 0
        updateCamPos(camera)
        camera.get_frame()
        img = camera.get_frame()
        corners, markerIds, rejectedCandidates = detector.detectMarkers(img)
        s = corners[0][0, 0]
        s0 = np.reshape(np.array(corners[0][0]), (8, 1))
        s0 = np.array([(ss - IMG_HALF) / IMG_HALF for ss in s0])
        s0_x = s0[::2]
        s0_z = s0[1::2]
        logs0_x.append(s0_x)
        logs0_z.append(s0_z)
        L0 = computeInterMatrix(Y0, s0)
        L0T = np.linalg.inv(L0.T @ L0) @ L0.T
        e = s0 - sd0
        coef = np.diag([1, 1, 1.5])
        w = -coef @ L0T @ e
    jStates = p.getJointStates(boxId, jointIndices=jointIndices)
    jPos = [state[0] for state in jStates]
    jVel = [state[1] for state in jStates]
    q1, q2 = jPos[0], jPos[1]
    vel1, vel2, vel3 = jVel[0], jVel[1], jVel[2]
    logPosf.append(jPos[2])
    logPosx.append(p.getLinkState(bodyUniqueId=boxId, linkIndex=6)[0][0])
    logPosz.append(p.getLinkState(bodyUniqueId=boxId, linkIndex=6)[0][2])
    J = np.array([[-L * np.cos(q1) - L * np.cos(q1 + q2), -L * np.cos(q1 + q2), 0],
                  [L * np.sin(q1) + L * np.sin(q1 + q2), L * np.sin(q1 + q2), 0],
                  [1, 1, 1]])

    dq = (np.linalg.inv(J) @ w).flatten()[[1, 0, 2]]
    dq[0] = -dq[0]
    dq[2] = -dq[2]

    C = np.array([[- 2 * m * L ** 2 * np.sin(q2) * vel1 * vel2 - m * L ** 2 * np.sin(q2) * vel2 ** 2],
                  [m * L ** 2 * vel1 ** 2 * np.sin(q2)],
                  [0]])
    G = np.array([[2 * m * g * L * np.sin(q1) + m * g * L * np.sin(q1 + q2)],
                  [m * g * L * np.sin(q1 + q2)],
                  [0]])
    M = np.array([[3 * m * L ** 2 + 2 * np.cos(q2) * m * L ** 2, m * L ** 2 + np.cos(q2) * m * L ** 2, 0],
                  [m * L ** 2 + np.cos(q2) * m * L ** 2, m * L ** 2, 0],
                  [0, 0, 6.666666666666666e-07]])
    u = C + G + np.array([[vel1], [vel2], [vel3]]) + M @ (k * -np.array([[vel1 - dq[0]], [vel2 - dq[1]], [vel3 - dq[2]]]))
    p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, controlMode=p.TORQUE_CONTROL,
                                forces=[u[0, 0], u[1, 0], u[2, 0]])
    #time.sleep(dt)
p.disconnect()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(logs0_x, logs0_z, s=2, label='trajecory')
plt.plot(sd0_x, sd0_z, 'r', linewidth=3, label='desired position')
logs0_x = np.concatenate((logs0_x[0], [logs0_x[0][0]]), axis=0)
logs0_z = np.concatenate((logs0_z[0], [logs0_z[0][0]]), axis=0)
plt.plot(logs0_x, logs0_z, linestyle='--', linewidth=3, label='starting position')
plt.legend(loc='upper right')
plt.xlabel('ImgX'), plt.ylabel('ImgZ')
plt.grid()

plt.subplot(3, 2, 2)
plt.plot(logTime, logPosx, 'k')
plt.axhline(y=xd, color='r', linestyle='--')
plt.title('X')
plt.grid()
plt.xlabel('t, s')
plt.ylabel('x, m')

plt.subplot(3, 2, 4)
plt.plot(logTime, logPosz, 'k')
plt.axhline(y=zd, color='r', linestyle='--')
plt.title('Z')
plt.grid()
plt.xlabel('t, s')
plt.ylabel('z, m')

plt.subplot(3, 2, 6)
plt.plot(logTime, logPosf, 'k')
plt.axhline(y=qd, color='r', linestyle='--')
plt.title(r'$\phi$ ')
plt.grid()
plt.xlabel('t, s')
plt.ylabel(r'$\phi$, rad')

plt.show()

