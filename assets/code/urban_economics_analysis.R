# Load Required Libraries
library(dplyr)
library(ggplot2)
library(gganimate)
library(sf)
library(maps)
library(ipumsr)

# Step 1: Load U.S. State Geometries from maps
us_states <- st_as_sf(map("state", plot = FALSE, fill = TRUE))
us_states <- us_states %>%
  mutate(name = tolower(ID))  # Convert state names to lowercase for joining

# Step 2: Data Preparation
# Load IPUMS microdata
ddi <- read_ipums_ddi("data/usa_00009.xml")
data <- read_ipums_micro(ddi)

# Calculate Total Population Aged 18-30 by State
total_pop <- data %>%
  filter(AGE >= 18 & AGE <= 30) %>%
  group_by(YEAR, STATEFIP) %>%
  summarise(total_pop = sum(PERWT, na.rm = TRUE))

# Calculate Divorced Population Aged 18-30 by State
divorced_pop <- data %>%
  filter(AGE >= 18 & AGE <= 30, MARST == 4) %>%
  group_by(YEAR, STATEFIP) %>%
  summarise(divorced_pop = sum(PERWT, na.rm = TRUE))

# Calculate Cohabiting Population Aged 18-30 by State
cohabiting_pop <- data %>%
  filter(AGE >= 18 & AGE <= 30, RELATED == 1114, MARST == 6) %>%
  group_by(YEAR, STATEFIP) %>%
  summarise(cohabiting_pop = sum(PERWT, na.rm = TRUE))

# Merge Datasets
merged_data <- total_pop %>%
  left_join(divorced_pop, by = c("YEAR", "STATEFIP")) %>%
  left_join(cohabiting_pop, by = c("YEAR", "STATEFIP"))

# Calculate Rates and Differential
final_data <- merged_data %>%
  mutate(
    divorce_rate = divorced_pop / total_pop,
    cohabitation_rate = cohabiting_pop / total_pop,
    differential = cohabitation_rate - divorce_rate
  ) %>%
  select(YEAR, STATEFIP, differential)

# Add State Names
state_names <- tibble(
  STATEFIP = c(1:51),
  name = tolower(c(
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming", "District of Columbia"
  ))
)

final_data <- final_data %>%
  left_join(state_names, by = "STATEFIP")

# Step 3: Merge Data with Geometries
us_map_data <- us_states %>%
  left_join(final_data, by = "name")

# Step 4: Create Static Choropleth Map
static_map <- ggplot(us_map_data) +
  geom_sf(aes(fill = differential, geometry = geometry), color = "white", size = 0.2) +
  scale_fill_gradient2(
    low = "blue", mid = "white", high = "red", midpoint = 0,
    name = "Differential"
  ) +
  labs(
    title = "Cohabitation vs Divorce Differential by State (Static View)",
    fill = "Differential"
  ) +
  theme_minimal() +
  theme(
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank()
  )


head(us_states)
head(final_data)
head(us_map_data)

# Create Animated Map
animated_map <- ggplot(us_map_data) +
  geom_sf(aes(fill = differential, geometry = geom), color = "white", size = 0.2) +
  scale_fill_gradient2(
    low = "blue", mid = "white", high = "red", midpoint = 0,
    name = "Differential"
  ) +
  # Step 1: Load U.S. State Geometries from maps
us_states <- st_as_sf(map("state", plot = FALSE, fill = TRUE))
us_states <- us_states %>%
  mutate(name = tolower(ID))  # Convert state names to lowercase for joining

# Step 2: Data Preparation
# Load IPUMS microdata
ddi <- read_ipums_ddi("data/usa_00009.xml")
data <- read_ipums_micro(ddi)

# Calculate Total Population Aged 18-30 by State
total_pop <- data %>%
  filter(AGE >= 18 & AGE <= 30) %>%
  group_by(YEAR, STATEFIP) %>%
  summarise(total_pop = sum(PERWT, na.rm = TRUE))

# Calculate Divorced Population Aged 18-30 by State
divorced_pop <- data %>%
  filter(AGE >= 18 & AGE <= 30, MARST == 4) %>%
  group_by(YEAR, STATEFIP) %>%
  summarise(divorced_pop = sum(PERWT, na.rm = TRUE))

# Calculate Cohabiting Population Aged 18-30 by State
cohabiting_pop <- data %>%
  filter(AGE >= 18 & AGE <= 30, RELATED == 1114, MARST == 6) %>%
  group_by(YEAR, STATEFIP) %>%
  summarise(cohabiting_pop = sum(PERWT, na.rm = TRUE))

# Merge Datasets
merged_data <- total_pop %>%
  left_join(divorced_pop, by = c("YEAR", "STATEFIP")) %>%
  left_join(cohabiting_pop, by = c("YEAR", "STATEFIP"))

# Calculate Rates and Differential
final_data <- merged_data %>%
  mutate(
    divorce_rate = divorced_pop / total_pop,
    cohabitation_rate = cohabiting_pop / total_pop,
    differential = cohabitation_rate - divorce_rate
  ) %>%
  select(YEAR, STATEFIP, differential)

# Add State Names
state_names <- tibble(
  STATEFIP = c(1:51),
  name = tolower(c(
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming", "District of Columbia"
  ))
)

final_data <- final_data %>%
  left_join(state_names, by = "STATEFIP")

# Step 3: Merge Data with Geometries
us_map_data <- us_states %>%
  left_join(final_data, by = "name")

# Step 4: Create Static Choropleth Map
static_map <- ggplot(us_map_data) +
  geom_sf(aes(fill = differential, geometry = geometry), color = "white", size = 0.2) +
  scale_fill_gradient2(
    low = "blue", mid = "white", high = "red", midpoint = 0,
    name = "Differential"
  ) +
  labs(
    title = "Cohabitation vs Divorce Differential by State (Static View)",
    fill = "Differential"
  ) +
  theme_minimal() +
  theme(
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank()
  )


head(us_states)
head(final_data)
head(us_map_data)

# Create Animated Map
animated_map <- ggplot(us_map_data) +
  geom_sf(aes(fill = differential, geometry = geom), color = "white", size = 0.2) +
  scale_fill_gradient2(
    low = "blue", mid = "white", high = "red", midpoint = 0,
    name = "Differential"
  ) + geom_sf_text(aes(label = name), size = 2, color = "black", fontface = "bold") +
  labs(
    title = "Cohabitation vs Divorce Differential by State (Year: {frame_time})",
    fill = "Differential"
  ) +
  theme_minimal() +
  theme(
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank()
  ) +
  transition_time(YEAR) +
  ease_aes('linear')

# Save as GIF
animate(
  animated_map,
  duration = 10, fps = 15, width = 800, height = 600,
  renderer = gifski_renderer("cohabitation_divorce_differential.gif")
)
  labs(
    title = "Cohabitation vs Divorce Differential by State (Year: {frame_time})",
    fill = "Differential"
  ) +
  theme_minimal() +
  theme(
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank()
  ) +
  transition_time(YEAR) +
  ease_aes('linear')

# Save as GIF
animate(
  animated_map,
  duration = 10, fps = 15, width = 800, height = 600,
  renderer = gifski_renderer("cohabitation_divorce_differential.gif")
)

