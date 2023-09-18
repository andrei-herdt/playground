$$
\begin{align}
\underset{x}{\text{minimise}} & \space \|Ax - x_{d}\|^2_{W_1} \\
\text{subject to} \space & \nonumber \\
& Cx + h \geq 0 \\
\end{align}
$$
For equality constrained problems, we can define the lagrangian:
$$
\begin{align}
\underset{x}{\text{minimise}} & \space \|Ax - x_{d}\|^2_{W_1} +  
\lambda (Cx + h)
\end{align}
$$
$$
\frac{\partial L}{\partial x} = 2A^T W_1 A x - 2A^T W_1 x_d + C^T \lambda = 0
$$
$$
\frac{\partial L}{\partial \lambda} = Cx + h = 0
$$
yielding the following system of equations:
$$
\begin{pmatrix}
2A^T W_1 A & C^T \\
C & 0
\end{pmatrix}
\begin{pmatrix}
x \\
\lambda
\end{pmatrix}
=
\begin{pmatrix}
2A^T W_1 x_d \\
- h
\end{pmatrix}
$$
