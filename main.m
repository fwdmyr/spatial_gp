clear; clc;
addpath(genpath(pwd));
data = load('PS111_CTD.mat');
data = data.PS111CTD;

doPlot = true;
KMTOM = 1E3;

% Closest event to Neumayer III is PS111_6-23
events = unique(data.Event);
latitudes = unique(data.Latitude);
longitude = unique(data.Longitude);

smallestInterval.begin = -inf;
smallestInterval.end = inf;

for i = 1:length(events)
    eventData = data(data.Event == events(i), :);
    minDepth = min(eventData.DepthWaterm);
    if (minDepth > smallestInterval.begin)
        smallestInterval.begin = minDepth;
    end
    maxDepth = max(eventData.DepthWaterm);
    if (maxDepth < smallestInterval.end)
        smallestInterval.end = maxDepth;
    end
    fprintf("Depth interval %s: [%.2f, %.2f]\n", events(i), minDepth, maxDepth);
end

smallestInterval.begin = ceil(smallestInterval.begin);
smallestInterval.end = floor(smallestInterval.end);

fprintf("Smallest depth interval: [%.2f, %.2f]\n", smallestInterval.begin, smallestInterval.end);

eventStrings = string(events);
eventStrings = arrayfun(@(event) strrep(event, '-', '_'), eventStrings);
eventIds = zeros(length(eventStrings), 1);
for i = 1:length(eventStrings)
    parts = strsplit(eventStrings(i), '_');
    eventIds(i) = double(parts(end));
end

% Sort events from closest to farthest wrt Neumayer III
[sortedEventIds, sortIdx] = sort(eventIds, 'descend');
sortedEvents = events(sortIdx);
sortedEventStrings = eventStrings(sortIdx);
sortedLatitudes = latitudes(sortIdx);
sortedLongitudes = longitude(sortIdx);

uniformData = struct;
uniformDepth = smallestInterval.begin:1:smallestInterval.end;

for i = 1:length(sortedEvents)
    eventData = data(data.Event == sortedEvents(i), :);
    uniformData(i).depth = uniformDepth;
    uniformData(i).temperature = interp1(eventData.DepthWaterm, eventData.TempC, uniformDepth); 
    uniformData(i).pressure = interp1(eventData.DepthWaterm, eventData.Pressdbar, uniformDepth); 
    uniformData(i).salinity = interp1(eventData.DepthWaterm, eventData.Sal, uniformDepth); 
end

[distRes, distLat, distLon] = pos2dist(sortedLatitudes(1), sortedLongitudes(1), sortedLatitudes(2), sortedLongitudes(2));
distLat = KMTOM * distLat;
distLon = KMTOM * distLon;
distRes = KMTOM * distRes;

obsDist = [0, distRes];
queryDist = 0:10:500;

origin = uniformData(1);
point = uniformData(2);

mapTemperature3D = zeros(length(queryDist), length(queryDist), length(uniformDepth));
mapSalinity3D = zeros(length(queryDist), length(queryDist), length(uniformDepth));
mapPressure3D = zeros(length(queryDist), length(queryDist), length(uniformDepth));

for i = 1:length(uniformDepth)
    obsTemperature = [origin.temperature(i), point.temperature(i)];
    planeTemperature = interp1(obsDist, obsTemperature, queryDist);

    obsSalinity = [origin.salinity(i), point.salinity(i)];
    planeSalinity = interp1(obsDist, obsSalinity, queryDist);

    obsPressure = [origin.pressure(i), point.pressure(i)];
    planePressure = interp1(obsDist, obsPressure, queryDist);

    for j = 1:length(queryDist)
        mapTemperature3D(j, :, i) = planeTemperature;
        mapSalinity3D(j, :, i) = planeSalinity;
        mapPressure3D(j, :, i) = planePressure;
    end
end

% Apply pseudo Perlin-Noise to the 3D maps
for i = 1:length(uniformDepth)
    mapTemperature3D(:, :, i) = perlinNoise(mapTemperature3D(:, :, i), 0.1);
    mapSalinity3D(:, :, i) = perlinNoise(mapSalinity3D(:, :, i), 1.0);
    mapPressure3D(:, :, i) = perlinNoise(mapPressure3D(:, :, i), 1E-5);
end

envData.temperature = mapTemperature3D;
envData.salinity = mapSalinity3D;
envData.pressure = mapPressure3D;
envData.N = queryDist;
envData.E = queryDist;
envData.D = uniformDepth;

save('EnvironmentMappingCTD.mat', 'envData');

if (doPlot)
    
    for i = 1:1:5

        figure;
        surf(mapTemperature3D(:, :, i));
        title('Temperature')
    
        figure;
        surf(mapSalinity3D(:, :, i));
        title('Salinity')
    
        figure;
        surf(mapPressure3D(:, :, i));
        title('Pressure')

    end

end

function grid = perlinNoise(grid, scaler)

    [n, m] = size(grid);
    i = 0;
    w = sqrt(n*m);

    while w > 5
        i = i + 1;
        d = interp2(scaler * randn(n, m), i-1, 'spline');
        grid = grid + i * d(1:n, 1:m);
        w = w - ceil(w/2 - 1);
    end
end