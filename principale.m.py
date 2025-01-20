import numpy as np
import matplotlib.pyplot as plt

#paramètres du modèle
T_e=1
T=100
sigma_Q=1
sigma_py=30
sigma_px=30
k=1000

#paramètres du filtre
F=np.array([[1,T_e,0,0],[0,1,0,0],[0,0,1,T_e],[0,0,0,1]])
H=np.array([[1,0,0,0],[0,0,1,0]])
Q=np.array([[T_e**3/3,T_e**2/2,0,0],[T_e**2/2,T_e,0,0],[0,0,T_e**3/3,T_e**2/2],[0,0,T_e**2/2,T_e]])*sigma_Q**2
R=np.array([[sigma_px**2,0],[0,sigma_py**2]])

x_init=np.transpose(np.array([3,40,-4,20]))
x_kalm=x_init
P_kalm=np.eye(4)


def creer_trajectoire(T,F,x_init,Q):
    x=np.zeros((4,T))
    x[:,0]=x_init
    for i in range(1,T):
        x[:,i]=np.dot(F,x[:,i-1])+np.random.multivariate_normal(np.zeros(4),Q)
    return x

vecteur_x=creer_trajectoire(T,F,x_init, Q)

def creer_observations(H,R,vecteur_x,T):
    y=np.zeros((2,T))
    for i in range(0,T):
        y[:,i]=np.dot(H,vecteur_x[:,i])+np.random.multivariate_normal(np.zeros(2),R)
    return y

vecteur_y=creer_observations(H,R,vecteur_x,T)

plt.plot(vecteur_x[0,:],vecteur_x[2,:],label='Trajectoire')
plt.plot(vecteur_y[0,:],vecteur_y[1,:],'ro',label='Observations')
plt.legend()
plt.show()


def filtre_de_Kalman(F,Q,H,R,y_k,x_kalm_prec,P_kalm_prec):
    #prediction
    x_pred=np.dot(F,x_kalm_prec)
    P_pred=np.dot(np.dot(F,P_kalm_prec),np.transpose(F))+Q
    if np.isnan(y_k).any():
        return x_pred, P_pred
    else:
    #mise à jour
        L=np.dot(H,np.dot(P_pred,np.transpose(H)))+R
        K=np.dot(np.dot(P_pred,np.transpose(H)),np.linalg.inv(L))
        y_tilde=y_k-np.dot(H,x_pred)
        x_kalm_k=np.dot(F,x_pred)+np.dot(K,y_tilde)
        P_kalm_k=P_pred-np.dot(np.dot(K,np.dot(np.dot(H,P_pred),np.transpose(H))+R),np.transpose(K))
        return x_kalm_k,P_kalm_k

x_est = np.zeros((4, T))

# Initialisation des premières valeurs
x_est[:, 0] = x_init
P_kalm_prec = P_kalm

# Boucle pour estimer les états cachés à chaque instant
for k in range(1, T):
    x_kalm, P_kalm_prec = filtre_de_Kalman(F, Q, H, R, vecteur_y[:, k], x_est[:, k-1], P_kalm_prec)
    x_est[:, k] = x_kalm


def err_quadra(k):
    err_quadra = np.dot(np.transpose(vecteur_x[:, k] - x_est[:, k]), vecteur_x[:, k] - x_est[:, k])
    return err_quadra

err_moyenne = T**(-1) * np.sum([err_quadra(k) for k in range(T)])

# Tracer la position vraie, estimée et observée en abscisse en fonction du temps
plt.figure()
plt.plot(range(T), vecteur_x[0, :], label='Position vraie (abscisse)')
plt.plot(range(T), x_est[0, :], label='Position estimée (abscisse)')
plt.plot(range(T), vecteur_y[0, :], 'ro', label='Position observée (abscisse)')
plt.xlabel('Temps')
plt.ylabel('Position en abscisse')
plt.legend()
plt.title('Position en abscisse en fonction du temps')
plt.show()

# Tracer la position vraie, estimée et observée en ordonnée en fonction du temps
plt.figure()
plt.plot(range(T), vecteur_x[2, :], label='Position vraie (ordonnée)')
plt.plot(range(T), x_est[2, :], label='Position estimée (ordonnée)')
plt.plot(range(T), vecteur_y[1, :], 'ro', label='Position observée (ordonnée)')
plt.xlabel('Temps')
plt.ylabel('Position en ordonnée')
plt.legend()
plt.title('Position en ordonnée en fonction du temps')
plt.show()

# Tracer la vraie trajectoire, la trajectoire observée et la trajectoire estimée par le filtre de Kalman
plt.figure()
plt.plot(vecteur_x[0, :], vecteur_x[2, :], label='Trajectoire vraie')
plt.plot(x_est[0, :], x_est[2, :], label='Trajectoire estimée')
plt.plot(vecteur_y[0, :], vecteur_y[1, :], 'ro', label='Trajectoire observée')
plt.xlabel('Position en abscisse')
plt.ylabel('Position en ordonnée')
plt.legend()
plt.title('Trajectoires')
plt.show()

# Charger les vecteurs d'observations pour les deux types d'avions
vecteur_y_avion_ligne = np.load('data/vecteur_y_avion_ligne.npy')
vecteur_y_avion_voltige = np.load('data/vecteur_y_avion_voltige.npy')
vecteur_x_avion_ligne = np.transpose(np.load('data/vecteur_x_avion_ligne.npy'))
vecteur_x_avion_voltige = np.transpose(np.load('data/vecteur_x_avion_voltige.npy'))

# Initialiser les états et les matrices de covariance pour chaque type d'avion
x_init_ligne = vecteur_x_avion_ligne[:, 0]
x_init_voltige = vecteur_x_avion_voltige[:, 0]

T_ligne = vecteur_y_avion_ligne.shape[0]
T_voltige = vecteur_y_avion_voltige.shape[0]

x_est_ligne = np.zeros((4, T_ligne))
x_est_voltige = np.zeros((4, T_voltige))

x_est_ligne[:, 0] = x_init_ligne
x_est_voltige[:, 0] = x_init_voltige

P_kalm_prec_ligne = P_kalm
P_kalm_prec_voltige = P_kalm

# Estimer les états cachés pour l'avion de ligne
for k in range(1, T_ligne):
    x_kalm, P_kalm_prec_ligne = filtre_de_Kalman(F, Q, H, R, vecteur_y_avion_ligne[k, :], x_est_ligne[:, k-1], P_kalm_prec_ligne)
    x_est_ligne[:, k] = x_kalm

# Estimer les états cachés pour l'avion de voltige
for k in range(1, T_voltige):
    x_kalm, P_kalm_prec_voltige = filtre_de_Kalman(F, Q, H, R, vecteur_y_avion_voltige[k, :], x_est_voltige[:, k-1], P_kalm_prec_voltige)
    x_est_voltige[:, k] = x_kalm

# Tracer les trajectoires estimées pour l'avion de ligne
plt.figure()
plt.plot(vecteur_x_avion_ligne[0, :], vecteur_x_avion_ligne[2, :], label='Trajectoire vraie (ligne)')
plt.plot(x_est_ligne[0, :], x_est_ligne[2, :], label='Trajectoire estimée (ligne)')
plt.plot(vecteur_y_avion_ligne[:, 0], vecteur_y_avion_ligne[:, 1], 'ro', label='Trajectoire observée (ligne)')
plt.xlabel('Position en abscisse')
plt.ylabel('Position en ordonnée')
plt.legend()
plt.title('Trajectoire de l\'avion de ligne')
plt.show()

# Tracer les trajectoires estimées pour l'avion de voltige
plt.figure()
plt.plot(vecteur_x_avion_voltige[0, :], vecteur_x_avion_voltige[2, :], label='Trajectoire vraie (voltige)')
plt.plot(x_est_voltige[0, :], x_est_voltige[2, :], label='Trajectoire estimée (voltige)')
plt.plot(vecteur_y_avion_voltige[:, 0], vecteur_y_avion_voltige[:, 1], 'ro', label='Trajectoire observée (voltige)')
plt.xlabel('Position en abscisse')
plt.ylabel('Position en ordonnée')
plt.legend()
plt.title('Trajectoire de l\'avion de voltige')
plt.show()

def mse(vecteur_x, x_est, T):
    return T**(-1) * np.sum([np.dot(np.transpose(vecteur_x[:, k] - x_est[:, k]), vecteur_x[:, k] - x_est[:, k]) for k in range(T)])

mse_ligne = mse(vecteur_x_avion_ligne, x_est_ligne, T_ligne)
mse_voltige = mse(vecteur_x_avion_voltige, x_est_voltige, T_voltige)

# Print the MSE values
print(f"MSE for airliner: {mse_ligne}")
print(f"MSE for aerobatic plane: {mse_voltige}")

# Visualize the MSE results using a bar chart
labels = ['Avion de ligne', 'Avion de voltige']
mse_values = [mse_ligne, mse_voltige]

plt.figure()
plt.bar(labels, mse_values, color=['blue', 'orange'])
plt.ylabel('Erreur moyenne')
plt.title('Erreur moyenne en fonction de lavion')
plt.show()