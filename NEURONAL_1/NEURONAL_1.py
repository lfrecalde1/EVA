from controller import Robot
import cv2 as cv
import numpy as np
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from numpy.linalg import inv
## tiempo sampleo de la simulacion
TIME_STEP = 100
robot = Robot()
## parte de los sensores de distancia

##acceso a los nodos de vision del robot
camera_id=[]    
camara=['camera','camera(1)']
for i in range(2):
    camera_id.append(robot.getCamera(camara[i]))
    camera_id[i].enable(TIME_STEP)

k_1=0
es=3
# lectura de los parametros de la matrix de la camara
cv_file = cv.FileStorage("fer_camara_3.yaml", cv.FILE_STORAGE_READ)
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()

## datos linea recta 1
recta1y=[]
recta1x=[]
recta2y=[]
recta2x=[]
distancia1=[]
distancia2=[]
## parametros para la red neuronal
net = cv.dnn.readNet('yolov3_training_last.weights', 'yolov3_testing.cfg')
classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

font = cv.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
## valores para poder grabar el video
fourcc = cv.VideoWriter_fourcc('D','I','V','3')

out = cv.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
## bucle de simulacion de robot

while robot.step(TIME_STEP) != -1:
    if k_1>es:
        k=cv.waitKey(1)
        ## obtencion de la imagen de la camara izquierda
        image_l = camera_id[1].getImageArray()
        left=np.uint8(image_l)
        left=cv.rotate(left,cv.ROTATE_90_CLOCKWISE)
        b_l,g_l,r_l=cv.split(left)
        ## imagen final para imagen lista de la macara izquierda
        procesamiento_l=cv.merge([r_l,g_l,b_l])
        procesamiento_l=cv.flip(procesamiento_l,1)
        ## procesamiento neuronal
        img = np.copy(procesamiento_l)
        height, width, _ = img.shape

        blob = cv.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []
        centros_finales=[]
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    centros_finales.append([center_x,center_y])
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
        ubicacion=np.array(centros_finales)
        
        if len(indexes)>0:
            for i in indexes.flatten():
                if(confidences[i]>0.2):
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i],2))
                    color = colors[i]
                    cv.rectangle(img, (x,y), (x+w, y+h), color, 2)
                    #cv.rectangle(procesamiento_l, (x,y), (x+w, y+h), color, 2)
                    cv.putText(img, label + " " + confidence, (x, y), font, 1, (255,255,255), 1)
        cv.imshow('Image', img)
        ##procesamiento para aruco
        gray_l = cv.cvtColor(procesamiento_l, cv.COLOR_BGR2GRAY)
        
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        
        parameters = aruco.DetectorParameters_create()
        
        parameters.adaptiveThreshConstant = 10
        
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_l, aruco_dict, parameters=parameters)
        datos_esquinas=np.array(corners)
        # font for displaying text (below)
        font = cv.FONT_HERSHEY_SIMPLEX
        if np.all(ids !=None):
            rvec, tvec ,_=aruco.estimatePoseSingleMarkers(corners, 0.1, mtx, dist)
            for i in range(0,ids.size):
                aruco.drawAxis(procesamiento_l, mtx, dist, rvec[i], tvec[i], 0.05)
            aruco.drawDetectedMarkers(procesamiento_l, corners)
            strg = ''
            for i in range(0,ids.size):
                strg += str(ids[i][0])+', '
            ## poscion de los marcadres ya que cambian de posicion alteatoriamente    
            posicion_1=np.array(np.where(ids==0))
            posicion_3=np.array(np.where(ids==1))
            posicion_4=np.array(np.where(ids==2))
            

            if (ubicacion.size==0)==True:
                cv.putText(procesamiento_l, "PERDIDA DE MARCADOR", (0,450), font, 1, (0,0,0),1,cv.LINE_AA)
            else:
                ## elemetos para graficar el marcador en el plano d ela imagen 
                punto_0=np.zeros((4,2),dtype=int)
                punto_0[0][0]=422
                punto_0[0][1]=363
               
                prueba1=np.array([[center_x,center_y]])
                punto_11=prueba1
                
                punto_1=np.zeros((4,2),dtype=int)
                punto_1[0][0]=472
                punto_1[0][1]=316
              

                punto_2=np.zeros((4,2),dtype=int)
                punto_2[0][0]=302
                punto_2[0][1]=324
                

                cv.circle(procesamiento_l, tuple(punto_0[0][:]), 5, (255,255,255), -1)
                cv.circle(procesamiento_l, tuple(punto_1[0][:]), 5, (0,255,255), -1)
                cv.circle(procesamiento_l, tuple(punto_2[0][:]), 5, (255,255,255), -1)

                ## calculo de la pendiente de las lineas
                fit = np.polyfit((punto_0[0][0],punto_2[0][0]), (punto_0[0][1],punto_2[0][1]), 1)
                fit1= np.polyfit((punto_0[0][0],punto_1[0][0]), (punto_0[0][1],punto_1[0][1]), 1)
                ## proyeccions de ambas lineas
                proyeccion_x=120
                proyeccion_y=fit[0]*proyeccion_x+fit[1]
                proyeccion_x_2=600
                proyeccion_y_2=fit1[0]*proyeccion_x_2+fit1[1]

                ## dibujo circulos de ambas lineas
                cv.circle(procesamiento_l, tuple([proyeccion_x,int(proyeccion_y)]), 5, (0,0,255), -1)
                cv.line(procesamiento_l, tuple(punto_0[0][:]), tuple([proyeccion_x,int(proyeccion_y)]), (0, 0, 0),3)

                cv.circle(procesamiento_l, tuple([proyeccion_x_2,int(proyeccion_y_2)]), 5, (0,0,255), -1)
                cv.line(procesamiento_l, tuple(punto_0[0][:]), tuple([proyeccion_x_2,int(proyeccion_y_2)]), (0, 0, 0),3)
                
                ## generacion de la recta 1
                for j in range(int(proyeccion_x),int(punto_0[0][0])):
                    recta1y.append(fit[0]*j+fit[1])
                    recta1x.append(j)
                
                recta1_final=np.array([recta1x,recta1y])
                ## generacion de la recta 2
                for j in range(int(punto_0[0][0]),int(proyeccion_x_2)):
                    recta2y.append(fit1[0]*j+fit1[1])
                    recta2x.append(j)
                
                recta2_final=np.array([recta2x,recta2y])
                ## minimos linea lateral
                almacenamieto1=np.zeros([ubicacion.shape[0],recta1_final.shape[1]])
                minimos1=[]

                for i in range(0,ubicacion.shape[0]):
                    for j in range(0,recta1_final.shape[1]):
                        d1=np.linalg.norm(np.array([recta1_final[0][j]-ubicacion[i][0],recta1_final[1][j]-ubicacion[i][1]]))
                        almacenamieto1[i][j]=d1
                
                for datos in almacenamieto1:
                    minimo_1_total = np.where(datos == np.amin(datos))
                    minimos1.append([minimo_1_total[0]])
                
                minimos1_final=np.array(minimos1).reshape(ubicacion.shape[0],1)
                
                for i in range(0,ubicacion.shape[0]):
                    cv.line(procesamiento_l, (ubicacion[i][0],ubicacion[i][1]),(recta1_final[0][minimos1_final[i]],recta1_final[1][minimos1_final[i]]),(0, 255, 0),2)
                ##--------------------------------minimos linea frontal
                almacenamieto2=np.zeros([ubicacion.shape[0],recta2_final.shape[1]])
                minimos2=[]

                for i in range(0,ubicacion.shape[0]):
                    for j in range(0,recta2_final.shape[1]):
                        d2=np.linalg.norm(np.array([recta2_final[0][j]-ubicacion[i][0],recta2_final[1][j]-ubicacion[i][1]]))
                        almacenamieto2[i][j]=d2
                
                for datos in almacenamieto2:
                    minimo_2_total = np.where(datos == np.amin(datos))
                    minimos2.append([minimo_2_total[0]])
                minimos2_final=np.array(minimos2).reshape(ubicacion.shape[0],1)
                
                for i in range(0,ubicacion.shape[0]):
                    cv.line(procesamiento_l, (ubicacion[i][0],ubicacion[i][1]),(recta2_final[0][minimos2_final[i]],recta2_final[1][minimos2_final[i]]),(255, 255, 0),2)
                    cv.putText(procesamiento_l,"rueda "+str(i),(ubicacion[i][0],ubicacion[i][1]) , font, 1, (255,0,0),1,cv.LINE_AA)
                
                for i in range(0,ubicacion.shape[0]):
                    err_x_linea1=-(recta1_final[0][minimos1_final[i]]-ubicacion[i][0])
                    err_y_linea1=recta1_final[1][minimos1_final[i]]-ubicacion[i][1]
                    err_x_linea2=(recta2_final[0][minimos2_final[i]]-ubicacion[i][0])
                    err_y_linea2=recta2_final[1][minimos2_final[i]]-ubicacion[i][1]
                    print(recta2_final[0][minimos2_final[i]])
                    if (err_x_linea1>0 and err_y_linea1>0 and err_x_linea2>0 and err_y_linea2>0):
                        cv.putText(procesamiento_l, "DENTRO DEL AREA", (0,430), font, 1, (0,0,0),1,cv.LINE_AA)
                        if(err_x_linea2<=20):
                            cv.putText(procesamiento_l, "ERROR FRONTAL EN "+str(i), (0,460), font, 1, (0,0,0),1,cv.LINE_AA)
                        if(err_y_linea1<=43):
                            
                            cv.putText(procesamiento_l, "ERROR LATERAL EN "+str(i), (0,490), font, 1, (0,0,0),1,cv.LINE_AA)
                    else:
                        cv.putText(procesamiento_l, "FUERA DEL AREA", (0,430), font, 1, (0,0,0),1,cv.LINE_AA)
            
        else:
            cv.putText(procesamiento_l, "No Ids", (0,64), font, 1, (0,255,0),1,cv.LINE_AA)


        ##muestra de datos de las camaras
        cv.imshow('left_1',procesamiento_l)
        if k%256==27:
            print("Escape hit, clossing")
            #file.close()
            break
    recta1y=[]
    recta1x=[]
    distancia1=[]
    recta2y=[]
    recta2x=[]
    distancia2=[]
    marcadores=[]
    k_1=k_1+1
cv.destroyAllWindows() 