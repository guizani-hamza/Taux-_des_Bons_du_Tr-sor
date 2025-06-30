
# Analyse des Taux des Bons du Trésor à 10 Ans

# Installation des packages nécessaires

# install.packages("forecast")
# install.packages("fredr")
# install.packages("tidyverse")
# install.packages("lubridate")
# install.packages("tseries")
# install.packages("zoo")
# install.packages("dygraphs")
# install.packages("ggplot2")
# install.packages("dplyr")

# Chargement des bibliothèques nécessaires
library(tidyverse)    # Manipulation de données
library(lubridate)    # Gestion des dates
library(zoo)          # Interpolation et manipulation des séries temporelles
library(forecast)     # Modélisation des séries temporelles
library(tseries)      # Tests statistiques pour séries temporelles
library(fredr)        # Accès à l'API FRED
library(dygraphs)     # Visualisation interactive
library(ggplot2)      # Graphiques
library(dplyr)        # Manipulation de données

# Configuration de la clé pour l'API FRED
fredr_set_key("fded5e5cfee46964aad695ca7288a937")

# Téléchargement des données via l'API FRED
data <- fredr(
  series_id = "DGS10",                       # Série FRED : taux des bons du Trésor à 10 ans
  observation_start = as.Date("1990-01-01"), # Date de début
  observation_end = as.Date("2020-12-31")    # Date de fin
)

# Vérification des données téléchargées
head(data)
str(data)
summary(data)

# Préparation des données

# Conversion de la colonne `date` au format Date
data$date <- as.Date(data$date)

# Tri des données et suppression des doublons
data <- data %>%
  distinct() %>%
  arrange(date)

# Visualisation de la série temporelle
ggplot(data, aes(x = date, y = value)) +
  geom_line(color = "blue") +
  labs(title = "Taux des Bons du Trésor à 10 ans", x = "Date", y = "Taux") +
  theme_minimal()

# Visualisation interactive
dygraph(data)

# Détection des valeurs manquantes
cat("Nombre de valeurs manquantes :", sum(is.na(data$value)), "\n")

# Interpolation des données manquantes
data$value <- zoo::na.approx(data$value)

# Vérification après interpolation
cat("Valeurs manquantes après interpolation :", sum(is.na(data$value)), "\n")

# Détection des valeurs aberrantes
iqr <- IQR(data$value, na.rm = TRUE)
lower_bound <- quantile(data$value, 0.25) - 1.5 * iqr
upper_bound <- quantile(data$value, 0.75) + 1.5 * iqr

# Filtrage des valeurs aberrantes
data <- data %>%
  filter(data$value >= lower_bound & data$value <= upper_bound)

boxplot(data$value, main = "Détection des valeurs aberrantes")

# Conversion des données en série temporelle
ts_data <- ts(data$value, frequency = 252, 
              start = c(year(min(data$date)), yday(min(data$date)) / 365))

# Visualisation de la série temporelle
plot(ts_data, main = "Série Temporelle du Taux des Bons du Trésor à 10 ans",
     ylab = "Taux", xlab = "Temps")

# Décomposition de la série temporelle
decomposed <- decompose(ts_data)
plot(decomposed)

# Autocorrélation
acf(ts_data, main = "Autocorrélation")
pacf(ts_data, main = "Autocorrélation partielle", lag.max = 36, lwd = 1.5)

# Test Dickey-Fuller augmentée
adf.test(ts_data, alternative = "stationary", k = 0)

# Division des données (80% entraînement, 20% test)
train_size <- floor(0.8 * length(ts_data))
train <- window(ts_data, end = c(2017, 101))  # Ajustement basé sur les données
test <- window(ts_data, start = c(2017, 102))

# Visualisation des ensembles
plot(ts_data, main = "Train vs Test Series", col = "black", lwd = 2)
lines(train, col = "blue", lwd = 2)
lines(test, col = "red", lwd = 2)
legend("topright", legend = c("Train", "Test"), col = c("blue", "red"), lty = 1, lwd = 2)

# Modèles simples

# Modèle moyen
mean_model <- meanf(train, h = length(test))

# Modèle naïf
naive_model <- naive(train, h = length(test))

# Modèle à dérive
drift_model <- rwf(train, drift = TRUE, h = length(test))

# Visualisation des prédictions
autoplot(ts_data) +
  autolayer(mean_model$mean, series = "Mean Model", color = "green") +
  autolayer(naive_model$mean, series = "Naive Model", color = "orange") +
  autolayer(drift_model$mean, series = "Drift Model", color = "purple") +
  labs(title = "Prédictions des modèles simples", x = "Temps", y = "Taux") +
  theme_minimal()

# Modèles avancés

# Modèle ARIMA
auto_arima_model <- auto.arima(train, seasonal = TRUE)
summary(auto_arima_model)

# Modèle ETS
ets_model <- ets(train)
summary(ets_model)

# Validation des résidus
checkresiduals(auto_arima_model)
checkresiduals(ets_model)

# Prédictions des modèles avancés

# Prédictions ARIMA
forecast_arima <- forecast(auto_arima_model, h = length(test))

# Prédictions ETS
forecast_ets <- forecast(ets_model, h = length(test))

# Visualisation des prédictions
autoplot(forecast_arima) +
  autolayer(test, series = "Test Data", color = "red") +
  labs(title = "Prédictions ARIMA", x = "Temps", y = "Taux") +
  theme_minimal()

autoplot(forecast_ets) +
  autolayer(test, series = "Test Data", color = "red") +
  labs(title = "Prédictions ETS", x = "Temps", y = "Taux") +
  theme_minimal()

# Comparaison des modèles

# Calcul des métriques de performance
accuracy_arima <- accuracy(forecast_arima, test)
accuracy_ets <- accuracy(forecast_ets, test)

# Affichage des résultats
cat("Performance ARIMA :\n")
print(accuracy_arima)
cat("\nPerformance ETS :\n")
print(accuracy_ets)
