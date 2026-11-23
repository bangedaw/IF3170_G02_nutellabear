"""SMO dengan seleksi working set orde kedua (bonus 6). Kelas induk di svm.py tidak diubah.

Induk memilih pasangan pelanggar maksimal (orde pertama): i = argmax -y G pada I_up, j = argmin -y G pada I_low.
Di sini i tetap sama, tetapi j dipilih dari kandidat di I_low yang melanggar bersama i dengan meminimalkan -b^2 / a,
yaitu penurunan objektif dual bila hanya pasangan (i, j) yang diperbarui:
    b_ij = (-y_i G_i) - (-y_j G_j) > 0,   a_ij = K_ii + K_jj - 2 K_ij (tau bila <= 0).
Kriteria berhenti tetap celah KKT pasangan pelanggar maksimal, sama dengan induk.
R.-E. Fan, P.-H. Chen, C.-J. Lin (2005). Working Set Selection Using Second Order Information for Training Support
Vector Machines. Journal of Machine Learning Research 6:1889-1918, https://jmlr.org/papers/v6/fan05a.html.
"""
import numpy as np

from svm import SVM

TAU = 1e-12


def select_second_order(score, low, i, K_row_i, K_ii, diagK):
    """Indeks j di I_low dengan score_j < score_i yang meminimalkan -b^2/a; None bila tidak ada kandidat."""
    candidates = np.flatnonzero(low & (score < score[i]))
    if not len(candidates):
        return None
    b = score[i] - score[candidates]
    a = K_ii + diagK[candidates] - 2 * K_row_i[candidates]
    a = np.where(a > 0, a, TAU)
    return int(candidates[np.argmin(-(b * b) / a)])


class SVMWSS2(SVM):
    """SVM yang sama dengan induk; hanya pemilihan j pada SMO yang memakai informasi orde kedua."""

    def _smo(self, K, y):
        n, C = len(y), self.C
        alpha = np.zeros(n)
        G = -np.ones(n)
        diagK = np.diag(K).copy()
        self.converged_ = False
        for it in range(self.max_iter):
            up = ((y > 0) & (alpha < C)) | ((y < 0) & (alpha > 0))
            low = ((y > 0) & (alpha > 0)) | ((y < 0) & (alpha < C))
            score = -y * G
            i = np.flatnonzero(up)[np.argmax(score[up])]
            gap = score[i] - score[low].min()
            if gap < self.tol:
                self.converged_ = True
                break
            j = select_second_order(score, low, i, K[i], diagK[i], diagK)
            eta = max(diagK[i] + diagK[j] - 2 * K[i, j], TAU)
            t = min((score[i] - score[j]) / eta,
                    C - alpha[i] if y[i] > 0 else alpha[i],
                    alpha[j] if y[j] > 0 else C - alpha[j])
            alpha[i] += y[i] * t
            alpha[j] -= y[j] * t
            G += y * t * (K[:, i] - K[:, j])
        self.n_iter_ = it + 1
        self.kkt_gap_ = gap
        free = (alpha > 1e-8) & (alpha < C - 1e-8)
        self.b_ = (-y * G)[free].mean() if free.any() else (score[up].max() + score[low].min()) / 2
        return alpha
