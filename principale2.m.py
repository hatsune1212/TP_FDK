import numpy as np
import matplotlib.pyplot as plt

# parametres du modele
T_e = 1
T = 100
sigma_Q = 10
sigma_dist = 10
sigma_angle = np.pi / 180

# Filter parameters
F = np.array([[1, T_e, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, T_e],
              [0, 0, 0, 1]])

H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

Q = np.array([[T_e ** 3 / 3, T_e ** 2 / 2, 0, 0],
              [T_e ** 2 / 2, T_e, 0, 0],
              [0, 0, T_e ** 3 / 3, T_e ** 2 / 2],
              [0, 0, T_e ** 2 / 2, T_e]]) * sigma_Q ** 2

R = np.array([[sigma_dist ** 2, 0],
              [0, sigma_angle ** 2]])

# Initialization
P_kalm = np.diag([100, 10, 100, 10])

vecteur_y_avion_ligne = np.load('data/vecteur_y_avion_ligne.npy')
vecteur_y_avion_voltige = np.load('data/vecteur_y_avion_voltige.npy')
vecteur_x_avion_ligne = np.transpose(np.load('data/vecteur_x_avion_ligne.npy'))
vecteur_x_avion_voltige = np.transpose(np.load('data/vecteur_x_avion_voltige.npy'))

print(len(vecteur_y_avion_ligne))
print(np.shape(vecteur_x_avion_voltige))
print(len(vecteur_y_avion_voltige))


def creer_observations_radar(R, vecteur_x, T):
    y = np.zeros((2, T))
    for i in range(0, T):
        distance = np.sqrt(vecteur_x[0, i] ** 2 + vecteur_x[2, i] ** 2)
        angle = np.arctan2(vecteur_x[2, i], vecteur_x[0, i])
        bruit = np.random.multivariate_normal(np.zeros(2), R)
        y[:, i] = np.array([distance, angle]) + bruit
    return y


vecteur_y_radar_ligne = creer_observations_radar(R, vecteur_x_avion_ligne, T)
vecteur_y_radar_voltige = creer_observations_radar(R, vecteur_x_avion_voltige, T)


def filtre_de_Kalman_radar(F, Q, R, y_k, x_kalm_prec, P_kalm_prec):
    # Prediction
    x_pred = np.dot(F, x_kalm_prec)
    P_pred = np.dot(F, np.dot(P_kalm_prec, F.T)) + Q

    # si y_k est vide
    if np.isnan(y_k).any():
        return x_pred, P_pred

    # maj
    H = np.array([[x_pred[0]/np.sqrt(x_pred[0] ** 2 + x_pred[2] ** 2), 0,
                   x_pred[2]/np.sqrt(x_pred[0] ** 2 + x_pred[2] ** 2), 0],
                  [-x_pred[2]/(x_pred[0] ** 2 + x_pred[2] ** 2), 0, x_pred[0] / (x_pred[0] ** 2 + x_pred[2] ** 2),
                   0]])

    L = np.dot(H, np.dot(P_pred, H.T)) + R
    K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(L)))

    y_tilde = y_k - np.array([np.sqrt(x_pred[0] ** 2 + x_pred[2] ** 2), np.arctan2(x_pred[2], x_pred[0])])
    x_kalm_k = x_pred + np.dot(K, y_tilde)
    P_kalm_k = P_pred - np.dot(K, np.dot(L, K.T))

    return x_kalm_k, P_kalm_k


x_est_ligne = np.zeros((4, T))
x_est_voltige = np.zeros((4, T))

# initialisation
x_est_ligne[:, 0] = vecteur_x_avion_ligne[:, 0]
x_est_voltige[:, 0] = vecteur_x_avion_voltige[:, 0]
P_kalm_prec_ligne = P_kalm
P_kalm_prec_voltige = P_kalm

# estimations trajectoires
for k in range(1, T):
    x_kalm_ligne, P_kalm_prec_ligne = filtre_de_Kalman_radar(F, Q, R, vecteur_y_radar_ligne[:, k],
                                                             x_est_ligne[:, k - 1], P_kalm_prec_ligne)
    x_est_ligne[:, k] = x_kalm_ligne

for k in range(1, T):
    x_kalm_voltige, P_kalm_prec_voltige = filtre_de_Kalman_radar(F, Q, R, vecteur_y_radar_voltige[:, k],
                                                                 x_est_voltige[:, k - 1], P_kalm_prec_voltige)
    x_est_voltige[:, k] = x_kalm_voltige



def cartesienne(y):
    x = np.zeros((2, y.shape[1]))# Ensure dimensions align with y
    for i in range(y.shape[1]):
        distance = y[0, i]
        angle = y[1, i]
        x[0, i] = distance * np.cos(angle)  # X-coordinate
        x[1, i] = distance * np.sin(angle)  # Y-coordinate
    return x

# Plot les trajectoires
plt.figure(figsize=(10, 6))
plt.plot(x_est_voltige[0, :], x_est_voltige[2, :], label='trajectoire est (Line)', color='b')
plt.plot(vecteur_x_avion_voltige[0, :], vecteur_x_avion_voltige[2, :], label='vraie trajectoire (Line)', color='g',
         linestyle='--')
plt.scatter(cartesienne(vecteur_y_radar_voltige)[0,:], cartesienne(vecteur_y_radar_voltige)[1,:], label='observations radar (Line)', color='r',
            s=10)

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kalman Filter Radar (Voltige)')
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x_est_ligne[0, :], x_est_ligne[2, :], label='trajectoire est (Line)', color='b')
plt.plot(vecteur_x_avion_ligne[0, :], vecteur_x_avion_ligne[2, :], label='vraie trajectoire (Line)', color='g',
         linestyle='--')
plt.scatter(cartesienne(vecteur_y_radar_ligne)[0,:], cartesienne(vecteur_y_radar_ligne)[1,:], label='observations radar (Line)', color='r',
            s=10)

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kalman Filter Radar (ligne)')
plt.grid()
plt.show()