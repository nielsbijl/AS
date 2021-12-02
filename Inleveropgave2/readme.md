# Inleveropgave 1: Model-based Prediction and Control

![myimage-alt-tag](https://i.ibb.co/Fx6CnBh/matrix.png)

De positie werkt zoals gebruikelijk bij een grid. Links onder is (0, 0)
en rechts boven is (3, 3).

### 3.2 Value Iteration
`discount = 1`

`Route: [(2, 0), (2, 1), (1, 1), (1, 2), (1, 3), (2, 3), (3, 3)]`

`Value funtion:` <br>
[[38, 39, 40, 0] <br>
[37, 38, 39, 40]<br>
[36, 37, 36, 35]<br>
[0, 36, 35, 34]]<br>

`Policy:`<br>
[[→], [→], [→], [None]]<br>
[[↑, →], [↑], [↑], [↑]]<br>
[[↑, →], [↑], [←], [←]]<br>
[[None], [↑], [↑], [↑, ←]]<br>


### 3.3 Optioneel: Value Iteration in non-deterministic environments:
Ik heb dit geïmplementeerd. Wanneer je de doolhof.run functie aanroept moet je meegeven of de uitvoering deterministisch is of niet.<br>

`discount = 1`

`Route: [(2, 0), (3, 0), (3, 1), (3, 2), (3, 3)]`

`Value funtion:`<br>
[[-30.560330861339544, -21.020164660235924, -0.494070670235244, 0]<br>
[-30.191397277484942, -22.079583022130485, -2.503358746886522, 30.212457796019287]<br>
[-25.810382804483204, -22.033141214969337, -10.749956440390845, 15.409396406685607]<br>
[0, -18.28507386883685, -10.395121959656356, 10.93581071434708]]<br>

`Policy:`<br>
[[→], [→], [→], [None]]<br>
[[→], [→], [→], [↑]]<br>
[[↓], [→], [→], [↑]]<br>
[[None], [←], [→], [↑]]<br>