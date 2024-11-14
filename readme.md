# Objetivo
O objetivo deste trabalho é criar um sistema de rastreamento de volante de Fórmula 1 em tempo real, utilizando visão computacional, técnicas de detecção de bordas e reconstrução 3D. O sistema será capaz de estimar a rotação do volante e plotar esses dados em gráficos, fornecendo uma análise visual do comportamento do volante ao longo de uma volta.

# Metodologia
O projeto será realizado em cinco etapas principais:

### a. Detecção de Bordas
Aplicação de técnicas de detecção de bordas nas imagens do cockpit para identificar os contornos do volante.

### b. Reconstrução 3D
Utilizando triangulação e correspondência de pontos, será criado um modelo 3D do volante baseado nas imagens 2D.

### c. Rastreamento de Pose
O modelo 3D será alinhado com as imagens em tempo real, mantendo a pose durante a movimentação do volante.

### d. Estimativa de Ângulo
O sistema calculará a rotação do volante em relação a um eixo fixo em cada frame.

### e. Plotagem de Dados
Os dados de ângulo do volante serão plotados em um gráfico, com o eixo X representando o tempo e o eixo Y, o ângulo do volante.

# Link para video original
[Lando Norris’ Pole Lap | 2024 Spanish Grand Prix | Pirelli](https://www.youtube.com/watch?v=pY0XNHVU-b0&t=6s&ab_channel=FORMULA1)
