function [c] = soundspeed(posNED)
%SOUNDSPEED Returns soundspeed from NED position

    persistent envData;

    if (isempty(envData))
        data = load('EnvironmentMappingCTD.mat');
        envData = data.envData;
    end

    N = posNED(1);
    E = posNED(2);
    D = posNED(3);

    % For now, dont support extrapolation
    assert(N >= envData.N(1) && N <= envData.N(end), "N = " + N + " not in grid [" + envData.N(1) + ", " + envData.N(end) + "]");
    assert(E >= envData.E(1) && E <= envData.E(end), "E = " + E + " not in grid [" + envData.E(1) + ", " + envData.E(end) + "]");
    assert(D >= envData.D(1) && D <= envData.D(end), "D = " + D + " not in grid [" + envData.D(1) + ", " + envData.D(end) + "]");

    temperature = interp3(envData.N, envData.E, envData.D, envData.temperature, N, E, D);
    salinity = interp3(envData.N, envData.E, envData.D, envData.salinity, N, E, D);
    pressure = interp3(envData.N, envData.E, envData.D, envData.pressure, N, E, D);

    % INPUT:
    %  SA  =  Absolute Salinity                                                          [ g/kg ]
    %  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
    %  p   =  sea pressure                                                                    [ dbar ]
    c = gsw_sound_speed_CT_exact(salinity, temperature, pressure);

end

