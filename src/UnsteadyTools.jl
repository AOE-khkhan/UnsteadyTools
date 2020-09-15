module UnsTools
import FLOWUnsteady
import FLOWNoise
import BPM
# import FLOWVLM
# reload("FLOWUnsteady")
uns = FLOWUnsteady
vlm = uns.vlm
vpm = uns.vpm
gt = uns.gt
noise = FLOWNoise

import Dierckx
import JLD

using PyPlot
using LaTeXStrings

thisdirectory = splitdir(@__FILE__)[1]
include(joinpath(thisdirectory,"UnsFileTools.jl"))

"""
Input:

* `data` : same format as output of `ARLTools.readTab()` function

Optional: 

* `Re::Float64` : in the case only one Re is represented. In this case, "RECL", "RECD", and "RECM" keys are ignored.

Output:

* `aeroFunction(alpha, Re, M)` : function that accepts any alpha, M, and Re, and returns the interpolated aerodynamic coefficients

"""
function buildAeroFunction(data::Dict{String, Union{Array{Float64,1}, Array{Float64,2}}}; 
    Re = nothing, 
    separatefunctions = false, 
    cl_spl_s=0.0, 
    cd_spl_s=0.0, 
    cm_spl_s=0.0,
    cl_spl_k_M=1,
    cd_spl_k_M=1,
    cm_spl_k_M=1,
    cl_spl_k_alpha=2,
    cd_spl_k_alpha=1,
    cm_spl_k_alpha=1,
    # cl_spl_w=nothing,
    # cd_spl_w=nothing,
    # cm_spl_w=nothing
)
    # extract data
    αs_cl = data["AOACL"]
    Ms_cl = data["MACHCL"]
    cls = data["CLTAB"]
    αs_cd = data["AOACD"]
    Ms_cd = data["MACHCD"]
    cds = data["CDTAB"]
    αs_cm = data["AOACM"]
    Ms_cm = data["MACHCM"]
    cms = data["CMTAB"]
    if Re == nothing # isnothing(Re)
        Res_cl = data["RECL"]
        Res_cd = data["RECD"]
        Res_cm = data["RECM"]
        throw("Re interpolation not yet implemented.")
    else
        spl_cl = Dierckx.Spline2D(αs_cl[:], Ms_cl[:], cls; s=cl_spl_s, kx=cl_spl_k_alpha, ky=cl_spl_k_M)
        spl_cd = Dierckx.Spline2D(αs_cd[:], Ms_cd[:], cds; s=cd_spl_s, kx=cd_spl_k_alpha, ky=cd_spl_k_M)
        spl_cm = Dierckx.Spline2D(αs_cm[:], Ms_cm[:], cms; s=cm_spl_s, kx=cm_spl_k_alpha, ky=cm_spl_k_M)
    end

    if !(separatefunctions)
        aeroFunction(α, Re, M) = spl_cl(α, M), spl_cd(α, M), spl_cm(α, M)
        aeroFunction(α::Array{T,1}, Re, M::T) where T = spl_cl(α, M * ones(length(α))), spl_cd(α, M * ones(length(α))), spl_cm(α, M * ones(length(α)))
        aeroFunction(α::T, Re, M::Array{T,1}) where T = spl_cl(α * ones(length(M)), M), spl_cd(α * ones(length(M)), M), spl_cm(α * ones(length(M)), M)
        return aeroFunction
    else
        clFunction(α::Array{T,1}, Re, M::Array{T,1}) where T = spl_cl(α, M)
        clFunction(α::Array{T,1}, Re, M::T) where T = spl_cl(α, M * ones(length(α)))
        cdFunction(α::Array{T,1}, Re, M::Array{T,1}) where T = spl_cd(α, M)
        cdFunction(α::Array{T,1}, Re, M::T) where T = spl_cd(α, M * ones(length(α)))
        cmFunction(α::Array{T,1}, Re, M::Array{T,1}) where T = spl_cm(α, M)
        cmFunction(α::Array{T,1}, Re, M::T) where T = spl_cm(α, M * ones(length(α)))
        return clFunction, cdFunction, cmFunction
    end
end

"""
Updates an array of FLOWVLM.airfoilprep.Polar objects for a new RPM value

Inputs:

* polars : the array of Polar objects to be updated
* `rR` : an array of normalized radial locations
* `R` : tip radius of the rotor
* clFunction : results of `buildAeroFunction()`
* cdFunction : results of `buildAeroFunction()`
* cmFunction : results of `buildAeroFunction()`
* `RPM` : of the rotor; used to calculate M numbers

Optional Inputs:

* `nu` : Float defining the kinematic viscocity of air
* `a_sound` : speed of sound in m/s

Output:

* nothing

"""
function updatePolars!(polars, rR, R, clFunctions::Array{Function,1}, cdFunctions::Array{Function,1}, cmFunctions::Array{Function,1}, RPM; nu = 1.48e-5, a_sound = 343.0, initialalpha = nothing)
    if length(polars) != length(rR)
        throw("Lenghts of polars and radial locations rR are not consistent")
    end
    if length(polars) != length(clFunctions) || length(polars) != length(cdFunctions) || length(polars) != length(cmFunctions)
        throw("Lenghts of polars and aero function arrays are not consistent")
    end
    omega = RPM * 2*pi / 60
    for ip = 1:length(polars)
        M = omega * rR / a_sound * R
        polars[ip].init_alpha = initialalpha == nothing ? polars[ip].init_alpha : initialalpha
        polars[ip].init_cl .= clFunctions[ip](polars[ip].init_alpha, polars[ip].init_Re, M[ip])
        polars[ip].init_cd .= cdFunctions[ip](polars[ip].init_alpha, polars[ip].init_Re, M[ip])
        polars[ip].init_cm .= cmFunctions[ip](polars[ip].init_alpha, polars[ip].init_Re, M[ip])
        polars[ip].pyPolar = vlm.ap.prepy[:Polar](polars[ip].init_Re, polars[ip].init_alpha, polars[ip].init_cl, polars[ip].init_cd, polars[ip].init_cm)
    end

    return nothing
end

function updatePolars!(polars, rR, R, clFunction, cdFunction, cmFunction, RPM; nu = 1.48e-5, a_sound = 343.0, initialalpha = nothing)
    if length(polars) != length(rR)
        throw("Lenghts of polars and radial locations rR are not consistent")
    end
    omega = RPM * 2*pi / 60
    for ip = 1:length(polars)
        M = omega * rR / a_sound * R
        polars[ip].init_alpha = initialalpha == nothing ? polars[ip].init_alpha : initialalpha
        polars[ip].init_cl .= clFunction(polars[ip].init_alpha, polars[ip].init_Re, M[ip])
        polars[ip].init_cd .= cdFunction(polars[ip].init_alpha, polars[ip].init_Re, M[ip])
        polars[ip].init_cm .= cmFunction(polars[ip].init_alpha, polars[ip].init_Re, M[ip])
        polars[ip].pyPolar = vlm.ap.prepy[:Polar](polars[ip].init_Re, polars[ip].init_alpha, polars[ip].init_cl, polars[ip].init_cd, polars[ip].init_cm)
    end

    return nothing
end

function updatePolars!(rotor::vlm.Rotor, clFunction, cdFunction, cmFunction, RPM; nu = 1.48e-5, a_sound = 343.0)
    polars = rotor._polars
    rR = rotor._r
    R = rotor.rotorR
    updatePolars!(polars, rR, R, clFunction, cdFunction, cmFunction, RPM; nu = nu, a_sound = a_sound)
end

"Takes a vector of nondimensional control point locations and returns the vector r_lat_custom for use in the `uns.generate_rotor()` function. 

Both r_lat_custom and rcontrolpoints input are nondimensional with 0 at the hub and 1 at the tip."
function rcp2r_lat_custom(rcontrolpoints)
    if rcontrolpoints[end] > 1.0
        error("Desired nondimensional control points radial locations exceeded 1")
    elseif rcontrolpoints[1] <= 0
        error("First control point must be > 0")
    end

    r_lat_custom = Float64[]
    push!(r_lat_custom, 0.0) # first lattice point should be at 0.0
    for i=1:length(rcontrolpoints)-1
        push!(r_lat_custom, (rcontrolpoints[i] + rcontrolpoints[i+1]) / 2.0)
    end
    push!(r_lat_custom, 1.0) # final lattice point should be at 1.0

    return r_lat_custom
end

function buildRotor(;
    # solver options 
    xfoil=false,
    # output options 
    verbose=true,
    v_lvl=0,
    plot_disc=true,
    # input files 
    rotor_file = "crc3.csv",
    data_path = "/Users/randerson/Box/BYU/FLOWResearch/projects/unsteadyTransition/crc20/",
    # Rotor geometry
    pitch = 0.0,                         # (deg) collective pitch of blades,
    CW = false,                          # Clock-wise rotation
    n = 10,                              # Number of blade elements,
    r_lat_custom=[],                     # if empty, does nothing; otherwise, specifies lattice discretization of the rotor; must be the same length as n+1
    splinesmoothing = 0.001,
    # conditions
    RPM=5000,                            # if not nothing and xfoil == true, incorporates Mach values in XFOIL
    magVinf = 0.0,
    # air properties
    asound = 343.0,
    nu = 1.48e-5
    )

    # Read radius of this rotor and number of blades
    R, B = uns.read_rotor(rotor_file; data_path=data_path)[[1,3]]
    ReD = RPM * 2*pi/60.0 * R * 2*R / nu
    Matip = RPM == nothing ? 0.0 : R * RPM * 2*pi/60 / asound

    Vinf(X,t) = magVinf * [1.0,0.0,0.0] # (m/s) freestream velocity


    # ------------ SIMULATION SETUP --------------------------------------------
    # Generate rotor
    rotor = uns.generate_rotor(rotor_file; pitch=pitch,
                                            n=n, CW=CW, ReD=ReD, Matip=Matip,
                                            verbose=verbose, xfoil=xfoil,
                                            data_path=data_path,
                                            plot_disc=plot_disc,
                                            r_lat_custom=r_lat_custom,
                                            spline_s = splinesmoothing)

    return rotor
end

"""
customWing()

Inputs include:

* ysections::Array{Real,1} : indicates spanwise nondimensional coordinates defining sections within which lattices are evenly spaced

    * elements span from -0.5 to 0.5

* nsections::Array{Int,1} : indicates the number of lattices contained within each section

    * length must be length(ysections) - 1

"""
function customWing(b, AR, clen, twist, ysections, nsections)
    if ysections[1] != -0.5 || ysections[end] != 0.5
        error("ysections must span from -1.0 to 1.0")
    elseif length(nsections) != length(ysections) - 1
        error("length of ysections and nsections inconsistent")
    end
    c = b / AR
    # get port leading edge coordinates
    porty = -b/2
    portx = 0.0
    portz = 0.0
    portc = c * clen[end]
    porttheta = twist[end]
    # create left chord vlm.Wing object 
    thiswing = vlm.Wing(portx, porty, portz, portc, porttheta)
    # add chords from port to starboard
    for (i, y) in enumerate(ysections[2:end])
        newy = y * b
        newx = 0.0
        newz = 0.0
        newc = c
        newtwist = 0.0
        n = nsections[i]
        vlm.addchord(thiswing, newx, newy, newz, newc, newtwist, n; r= 1.0, central=false, refinement=[])
    end

    return thiswing
end

"""
Build the CRC-3 and return a FLOWUnsteady.VLMVehicle object
"""
function buildCRC3( ;
    # output options 
    today = "20200721",
    runname = "crc3geometry",
    savepath = joinpath(UnsFileTools.extdrive_path, today, runname),
    savevtk = true,
    verbose = false,
    plot_disc = true,
    # input options 
    datapath = "/Users/randerson/Box/BYU/FLOWResearch/projects/unsteadyTransition/crc20",

    O = zeros(3),
    aerodynamicsframe = false,
    b = 20 * 0.0254, # 1.497, # span
    c = 0.28717 * 0.3048, 
    lambda = 1.0, # taper ratio
    Lambda = 0.0, # sweep angle (radians)
    phi = 0.0, # dihedral
    thetar = 0.0, # root twist
    thetat = 0.0, # tip twist
    gamma = 0.0, # dihedral
    n = 50, # number of lattices
    wingsections = [
        -0.83333
        # -0.80780
        # -0.78227
        # -0.75674
        # -0.73121
        # -0.70568
        # -0.68015
        # -0.65462
        -0.62909
        # -0.55202
        # -0.47495
        # -0.39788
        # -0.32081
        # -0.24374
        -0.16667
        # -0.14114
        # -0.11561
        # -0.09008
        # -0.06455
        # -0.03902
        -0.01349
        # 0.0    
        0.01349
        # 0.03902
        # 0.06455
        # 0.09008
        # 0.11561
        # 0.14114
        0.16667
        # 0.24374
        # 0.32081
        # 0.39788
        # 0.47495
        # 0.55202
        0.62909
        # 0.65462
        # 0.68015
        # 0.70568
        # 0.73121
        # 0.75674
        # 0.78227
        # 0.80780
        0.83333
        ] / 0.83333 / 2,
    nsections = [
        8
        6
        6
        2
        6
        6
        8
        ],
    wing_spacing = 0.254, # m - distance between wings 
    qc_lex = 65.1925e-3, # m - distance from rotor hub to wing quarter chord

    initialpitch = 2.0 * pi/180,
    initialroll  = 0,
    initialyaw   = 0,
    # rotor options
    includerotors = true,
    # provide rotor objects 
    cw_rotor = nothing, # if this and `ccwrotor == nothing`, ignored; otherwise, specifies the FLOWVLM.Rotor object
    ccw_rotor = nothing,
    # build rotors from file 
    rotorfile = "crc3.csv", # ignored if cwrotor and ccwrotor != nothing
    initialrpm = 2000.0,
    rotorcollective=30.0,
    nbladeelements=10,
    r_lat = 0.5,
    # rotor position/orientation wrt vehicle
    rotor_y = 242.55e-3/2, # 0.365, # m
    rotor_inboardrotation = 10*pi/180,

    vehicletype = uns.VLMVehicle
)
    
    # set up directories 

    if !isdir(savepath)
        mkdir(savepath)
    end

    vehicleaxis = getAttitude(initialpitch, initialroll, initialyaw; aero2controls = !(aerodynamicsframe))

    unrotatedaxis = eye(3)

    # build wings
    pos = [0.0,1.0]
    clen = [1,1]
    twist = [0.0, 0.0]
    sweep = [0.0]
    dihed = [0.0]

    AR = b/c

    wing_lex = qc_lex - c/4 # assume rotors are at x=0
    # topwing = buildWing(O, b, AR, lambda, thetar, thetat, Lambda, gamma)
    topwing = customWing(b, AR, clen, twist, wingsections, nsections)
    vlm.setcoordsystem(topwing, unrotatedaxis * [0.0, 0.0, wing_spacing/2] + [wing_lex, 0.0, 0.0], unrotatedaxis)
    # bottomwing = vlm.complexWing(b, AR, n, pos, clen, twist, sweep, dihed; symmetric=true, chordalign=0, _ign1=false)
    bottomwing = customWing(b, AR, clen, twist, wingsections, nsections)
    vlm.setcoordsystem(bottomwing, unrotatedaxis * [0.0, 0.0, -wing_spacing/2] + [wing_lex, 0.0, 0.0], unrotatedaxis)
    wing_system = vlm.WingSystem()
    vlm.addwing(wing_system, "topwing", topwing)
    vlm.addwing(wing_system, "bottomwing", bottomwing)
    vlmsystem = vlm.WingSystem()
    vlm.addwing(vlmsystem, "wing_system", wing_system)

    # build rotors
    rotor_xs = ones(4) * -c/4
    rotor_ys = [rotor_y, -rotor_y, -rotor_y, rotor_y]
    rotor_zs = [-wing_spacing/2, -wing_spacing/2, wing_spacing/2, wing_spacing/2]
    cw = [false, true, false, true]

    # modified dimensions from:
    # property,file,description
    # Rtip,0.127, (m) Radius of blade tip
    # Rhub,0.0095325, (m) Radius of hub
    # B,2, Number of blades
    # blade,apc10x7_blade.csv, Blade file

    if includerotors
        # R, B = uns.read_rotor(rotorfile; data_path=datapath)[[1,3]] # radius and number of blades
        if cw_rotor == nothing; cw_rotor = uns.generate_rotor(rotorfile; pitch=rotorcollective,
            n=nbladeelements, r_lat=r_lat,
            CW=true, ReD=1.5e6,
            verbose=verbose, xfoil=false,
            data_path=datapath,
            plot_disc=plot_disc); end
        if ccw_rotor == nothing; ccw_rotor = uns.generate_rotor(rotorfile; pitch=rotorcollective,
            n=nbladeelements, r_lat=r_lat, 
            CW=false, ReD=1.5e6,
            verbose=verbose, xfoil=false,
            data_path=datapath,
            plot_disc=plot_disc); end

        rotors = vlm.Rotor[]
        testrotorsystem = vlm.WingSystem()
        signrotation = [1,-1,-1,1]

        for i=1:length(rotor_xs)
            # set cw/ccw
            thisrotor = cw[i] ? deepcopy(cw_rotor) : deepcopy(ccw_rotor)
            thisrotor.airfoils = cw[i] ? cw_rotor.airfoils : ccw_rotor.airfoils
            thisrotor._polars = cw[i] ? cw_rotor._polars : ccw_rotor._polars
            thisrotor._polarroot = cw[i] ? cw_rotor._polarroot : ccw_rotor._polarroot
            thisrotor._polartip = cw[i] ? cw_rotor._polartip : ccw_rotor._polartip
            # set location
            origin = [rotor_xs[i], rotor_ys[i], rotor_zs[i]]
            rotoraxis = [cos(rotor_inboardrotation) sin(rotor_inboardrotation)*signrotation[i] 0;
                        -sin(rotor_inboardrotation)*signrotation[i] cos(rotor_inboardrotation) 0;
                         0 0 1
                         ]
            vlm.setcoordsystem(thisrotor, origin, rotoraxis * unrotatedaxis; user=true)
            # initial rotation
            vlm.rotate(thisrotor, 90)
            # add to rotors
            push!(rotors, thisrotor)
            vlm.addwing(testrotorsystem, "rotor$(i)", thisrotor)
        end
        rotorsystems = (rotors, )
    else
        rotorsystems = ()
    end

    system = vlm.WingSystem()
    vlm.addwing(system, "Wings", wing_system)

    wakesystem = vlm.WingSystem()
    vlm.addwing(wakesystem, "SolveVLM", vlmsystem)
    if includerotors
        for (i, rotor) in enumerate(rotors)
            vlm.addwing(wakesystem, "Rotor$i", rotor)
            vlm.addwing(system, "Rotor$i", rotor)
        end
    end

    # rotate vehicle
    vlm.setcoordsystem(system, zeros(3), vehicleaxis)

    # TODO: must we add this now?
    # add freestream
    # Vinf(x,t) = [1.0,0.0,0.0]         #non-dimensional function defining free stream velocity
    # vlm.setVinf(system, Vinf)

    tiltingsystems = () # no tilting systems
    # vlm.save(system, runname; path=savepath, save_horseshoes=false)

    # build the vehicle
    vehicle = vehicletype(system;
        tilting_systems = tiltingsystems,
        rotor_systems = rotorsystems,
        vlm_system = vlmsystem,
        wake_system = wakesystem
        );

    for rotor_system in vehicle.rotor_systems
        for rotor in rotor_system
            vlm.setRPM(rotor, initialrpm)
        end
    end

    # create vtk rendering
    if savevtk
        strn = uns.save_vtk(vehicle, runname; path=savepath, save_horseshoes=false)
    end
    # strn = vlm.save(vehicle.rotor_systems[1][1], joinpath(savepath,runname*"test_this_prop"); save_horseshoes=false)
    # strn = uns.save_vtk(rotorsystems[1][1], runname*"test_this_prop"; path=savepath, save_horseshoes=false)
    # run(`paraview --data="$savepath/$strn"`)

    return vehicle
end

"""
Returns the reference frame with the specified `pitch`, `roll`, and `yaw` angles (given in radians), compatible with FLOWUnsteady.
"""
function getAttitude(pitch, roll, yaw; aero2controls = false)
    # set orientation
    rotatepitch = [
        cos(pitch) 0 -sin(pitch); 
        0.0 1.0 0.0; 
        sin(pitch) 0.0 cos(pitch)
    ]
    rotateroll = [
        1 0 0; 
        0.0 cos(roll) sin(roll); 
        0.0 -sin(roll) cos(roll)
    ]
    rotateyaw = [
        cos(yaw) sin(yaw) 0;
        -sin(yaw) cos(yaw) 0;
        0 0 1
    ]
    vehicleaxis = rotateyaw * rotatepitch * rotateroll # note this order matters

    if aero2controls
        vehicleaxis[1,:] *= -1.0
        vehicleaxis[3,:] *= -1.0
    end

    return vehicleaxis
end

function simulateVehicle(today, runname;
    # output options
    savepath = joinpath(UnsFileTools.extdrive_path,today,runname),
    visualizemaneuver = false,
    prompt = true,
    paraview = false,
    wingmonitor = true,
    rotormonitor = true,
    verbose = true, v_lvl = false,
    # vehicle options 
    vehicle = nothing,
    vehicleconstructor = buildCRC20,
    vehiclearguments = [],
    # initial vehicle orientation...
    # maneuver options 
    Vvehicle = (t) -> [0.0,0.0,0.0],    # Translational velocity of vehicle over Vcruise
    anglevehicle = (t) -> zeros(3),     # (deg) angle of the vehicle
    angle = (),
    RPMfunctions = ((t) -> 1.0, ),
    # simulation options
    tinit = 0.0, # initial clock time
    Vref = 0.0,
    RPMref = 5000.0,
    elapsedtime = 3.0, # seconds
    nsteps = 100,
    choptimes = [],
    choptypes = [],
    choplocations = [],
    chopreverses = [],
    # VPM parameters
    shed_unsteady = true,
    p_per_step = 1,
    vlm_sigma = -1,
    surf_sigma_factor = 1/10.0,
    asound_sim = nothing, # if nothing, omits PG Mach correction during the simulation; set this to false if airfoil tables contain Mach sensitive data already 
    overwrite_sigma = nothing,
    sigmafactor = 2.125,
    wake_coupled = true,
    # debug options
    save_rotor_v = false,
    nsteps_save_rotor = 10,
    nsteps_save_fig = 10,
    # freestream fluid properties
    Vinf = (x, t) -> zeros(3),
    rho = 1.225, # kg/m^3
    nu = 1.48e-5 # m^2/s
)

    # ----- PREPARE DIRECTORIES
    if !(isdir(joinpath(UnsFileTools.extdrive_path,today)))
        mkdir(joinpath(UnsFileTools.extdrive_path,today))
    end

    # ----- BUILD VEHICLE 
    if vehicle == nothing
        vehicle = vehicleconstructor(; vehiclearguments...)
    end

    # ----- GENERATE MANEUVER
    maneuver = uns.KinematicManeuver(angle, RPMfunctions, Vvehicle, anglevehicle)

    # ----- DEFINE SIMULATION
    Vinit = Vref*maneuver.Vvehicle(tinit/elapsedtime)   # (m/s) initial vehicle velocity
    angle1 = maneuver.anglevehicle(tinit/elapsedtime)
    angle2 = maneuver.anglevehicle(tinit/elapsedtime + 1e-12)
    Winit = pi/180 * (angle2-angle1)/((elapsedtime - tinit)*1e-12) # (rad/s) initial vehicle angular velocity

    simulation = uns.Simulation(vehicle, maneuver, Vref, RPMref, elapsedtime; Vinit=Vinit, Winit=Winit, t=tinit)

    # ----- DEFINE EXTRA RUNTIME FUNCTIONS
    rotorMonitor = getRotorMonitor(simulation, nsteps;
        save_path = savepath,
        run_name = runname,
        nsteps_savefig = nsteps_save_fig,
        save_pfield = false,
        nsteps_save_pfield = 100,
        save_jld = true,
        verbose = true, v_lvl = 0,
        choptimes = choptimes,
        choptypes = choptypes,
        choplocations = choplocations,
        chopreverses = chopreverses,
        rho = rho
    )

    wingMonitor = getWingMonitor(simulation, nsteps, Vinf;
        figname="monitor_wing", 
        nsteps_plot = 1,
        disp_plot = true, 
        figsize_factor = 1.0,
        save_path = savepath, 
        run_name = runname,
        nsteps_savefig = 10,
        extra_plots = true,
        magVinf = nothing,
        rhoinf = 1.225,
        wake_coupled = true
    )

    monitor(args...) = (rotormonitor ? rotorMonitor(args...) : false) || (wingmonitor ? wingMonitor(args...) : false)

    # ----- GET MAX NUMBER OF PARTICLES FOR PREALLOCATION
    maxparticles = maxParticles(simulation, p_per_step, nsteps)
    
    # ----- GET SURFACE PARTICLE RADIUS
    if !(isempty(vehicle.rotor_systems))
        R = vehicle.rotor_systems[1][1].rotorR
    else
        error("No rotors found; cannot define surf_sigma")
    end

    if visualizemaneuver
        strn = uns.visualize_kinematics(simulation, nsteps, savepath;
                            run_name=runname,
                            save_vtk_optsargs=[(:save_horseshoes, false)],
                            prompt=prompt, verbose=verbose, v_lvl=v_lvl,
                            paraview=paraview
                            )
        return strn, vehicle
    else
        pfield = uns.run_simulation(simulation, nsteps;
                            # SIMULATION OPTIONS
                            Vinf=Vinf,
                            # SOLVERS OPTIONS
                            p_per_step=p_per_step,
                            overwrite_sigma=overwrite_sigma,
                            sigmafactor=sigmafactor,
                            vlm_sigma=vlm_sigma,
                            surf_sigma=R * surf_sigma_factor,
                            max_particles=maxparticles,
                            shed_unsteady=shed_unsteady,
                            wake_coupled = wake_coupled,
                            extra_runtime_function=monitor,
                            # OUTPUT OPTIONS
                            save_path=savepath,
                            run_name=runname,
                            prompt=prompt,
                            verbose=verbose, v_lvl=v_lvl,
                            save_code=splitdir(@__FILE__)[1],
                            save_horseshoes=false,
                            sound_spd=asound_sim,         # we'll apply mach corrections manually
                            # save_rotor_v=save_rotor_v,        # whether to save rotor relative net swirl/axial velocity
                            # nsteps_save_rotor=nsteps_save_rotor       # save rotor velocity every this many steps
                            )
        return pfield, vehicle
    end
end

function maxParticles(simulation::uns.Simulation, p_per_step::Int, nsteps::Int)
    wake_system = simulation.vehicle.wake_system
    maxNP = 0
    for wing in wake_system.wings
        maxNP += maxParticles(wing, p_per_step, nsteps)
    end

    return maxNP
end

function maxParticles(wingsystem::vlm.WingSystem, p_per_step::Int, nsteps::Int)
    maxNP = 0
    for wing in wingsystem.wings
        maxNP += maxParticles(wing, p_per_step, nsteps)
    end

    return maxNP
end

function maxParticles(wing::vlm.Wing, p_per_step::Int, nsteps::Int)
    maxNP = (2 * wing.m + 1) * nsteps * p_per_step
end

function maxParticles(rotor::vlm.Rotor, p_per_step::Int, nsteps::Int)
    maxNP = (2 * rotor.m + 1) * rotor.B * nsteps * p_per_step
end

"""
getRotorMonitor(

)

Returns:

* a function of the form:

    * myMonitor(sim::uns.Simulation, pfield::vpm.ParticleField, t::Float64, dt::Float64)
    * returns `true` if the simulation breaks

"""
function getRotorMonitor(simulation, nsteps;
    save_path = splitdir(@__FILE__)[1],
    run_name = "defaultrunname",
    nsteps_savefig = 10,
    save_pfield = false,
    nsteps_save_pfield = 50,
    save_jld = true,
    verbose = true, v_lvl = 0, 
    plotme = true,
    figsize_factor = 1.0,
    choptimes = [],
    choptypes = [],
    choplocations = [],
    chopreverses = [],
    rho = 1.225
)
    # declare variables
    functioncalls = 0
    styles = "o^*.pxs"^100 # plot styles
    # rotor monitor variables
    nrotors = length(vcat(simulation.vehicle.rotor_systems...))
    CTs = zeros(nsteps+1, nrotors)
    CQs = zeros(nsteps+1, nrotors)
    ηs = zeros(nsteps+1, nrotors)
    Ts = zeros(nsteps+1,1)
    angles = zeros(nsteps+1,nrotors)
    # get vector of RPM functions corresponding to `vcat(simulation.vehicle.rotor_systems...)`
    RPMfunctions = Array{Function,1}()
    for (i, rotorsystem) in enumerate(simulation.vehicle.rotor_systems)
        for rotor in rotorsystem
            push!(RPMfunctions, simulation.maneuver.RPM[i])
        end
    end
    if length(RPMfunctions) != nrotors; throw("RPMfunctions and rotors length inconsistent"); end

    # declare monitor function
    function rotorMonitor(sim::uns.Simulation, pfield::vpm.ParticleField, T::Float64, DT::Float64; rho = 1.225)
        functioncalls += 1
        # plot rotor performance
        rotors = vcat(sim.vehicle.rotor_systems...)
        # update angles 
        if functioncalls > 1
            for rotori in 1:length(rotors)
                angles[functioncalls, rotori] = angles[functioncalls - 1, rotori] + integratetrap([T - DT, T] ./ simulation.ttot, RPMfunctions[rotori]; scaley = 360.0 / 60.0 * simulation.RPMref * simulation.ttot)
            end
        else
            angles[functioncalls, :] = zeros(nrotors)
        end
        # update Ts
        Ts[functioncalls] = T

        if functioncalls==1
            figure("rotorconvergenceplot", figsize=[7*3,5*2] * figsize_factor)
            subplot(231)
            title("Circulation Distribution")
            xlabel("Element index")
            ylabel(L"Circulation $\Gamma$ (m$^2$/s)")
            grid(true, color="0.8", linestyle="--")
            subplot(232)
            title("Plane-of-rotation Normal Force")
            xlabel("Element index")
            ylabel(L"Normal Force $N_p$ (N)")
            grid(true, color="0.8", linestyle="--")
            subplot(233)
            title("Plane-of-rotation Tangential Force")
            xlabel("Element index")
            ylabel(L"Tangential Force $T_p$ (N)")
            grid(true, color="0.8", linestyle="--")
            subplot(234)
            xlabel(L"Age $\psi$ ($^\circ$)")
            ylabel(L"Thrust $C_T$")
            grid(true, color="0.8", linestyle="--")
            subplot(235)
            xlabel(L"Age $\psi$ ($^\circ$)")
            ylabel(L"Torque $C_Q$")
            grid(true, color="0.8", linestyle="--")
            subplot(236)
            xlabel(L"Age $\psi$ ($^\circ$)")
            ylabel(L"Propulsive efficiency $\eta$")
            grid(true, color="0.8", linestyle="--")
        end
        
        for (i,rotor) in enumerate(rotors)
            CT, CQ = vlm.calc_thrust_torque_coeffs(rotor, rho)
            rotororientation = rotor._wingsystem.Oaxis[1,:]
            J = dot(rotororientation, simulation.maneuver.Vvehicle(T/simulation.ttot)) * # get component of the freestream perpendicular to the rotor plane
            simulation.Vref / rotor.RPM * 60 / 2 / rotor.rotorR
            eta = J*CT/(2*pi*CQ)
            CTs[functioncalls,i] = CT
            CQs[functioncalls,i] = CQ
            ηs[functioncalls,i] = eta
            
            if plotme
                # prepare plot colors/styles
                cratio = pfield.nt/nsteps
                cratio = cratio > 1 ? 1 : cratio
                clr = functioncalls==1 && false ? (0,0,0) : (1-cratio, 0, cratio)
                stl = functioncalls==1 && false ? "o" : "-"
                alpha = functioncalls==1 && false ? 1 : 0.5
                # prepare rotor plot
                figure("rotorconvergenceplot")
                # Circulation distribution
                subplot(231)
                this_sol = []
                # for rotor in rotors
                this_sol = vcat(this_sol, [vlm.get_blade(rotor, j).sol["Gamma"] for j in 1:rotor.B]...)
                # end
                plot(1:size(this_sol,1), this_sol, stl, alpha=alpha, color=clr)
    
                # Np distribution
                subplot(232)
                this_sol = []
                # for rotor in rotors
                this_sol = vcat(this_sol, rotor.sol["Np"]["field_data"]...)
                # end
                plot(1:size(this_sol,1), this_sol, stl, alpha=alpha, color=clr)
    
                # Tp distribution
                subplot(233)
                this_sol = []
                # for rotor in rotors
                this_sol = vcat(this_sol, rotor.sol["Tp"]["field_data"]...)
                # end
                plot(1:size(this_sol,1), this_sol, stl, alpha=alpha, color=clr)

                subplot(234)
                plot([angles[functioncalls, i]], [CT], "$(styles[i])", alpha=alpha, color=clr)
                subplot(235)
                plot([angles[functioncalls, i]], [CQ], "$(styles[i])", alpha=alpha, color=clr)
                subplot(236)
                plot([angles[functioncalls, i]], [eta], "$(styles[i])", alpha=alpha, color=clr)

                # Save figure
                if functioncalls%nsteps_savefig==0 && functioncalls!=1 && save_path!=nothing
                    savefig(joinpath(save_path, run_name*"_rotorconvergence.svg"),
                                                                transparent=false)
                end
            end
            # chop wake
            for chopi = 1:length(choptimes)
                if functioncalls == choptimes[chopi]
                    chopWake!(pfield, choptypes[chopi], choplocations[chopi]; rotors=rotors, verbose=verbose, v_lvl=v_lvl, reverselogic=chopreverses[chopi])
                end
            end
        end
        
        # save JLD
        if functioncalls%nsteps_savefig==0 && functioncalls!=1 && save_jld
            println("Saving JLD file as: ",run_name*"_TQeta.jld\n")
            JLD.save(joinpath(save_path, run_name*"_TQeta.jld"),
                "CTs",CTs,
                "CQs",CQs,
                "ηs",ηs,
                "Ts",Ts,
                "angles",angles
            )
        end

        # save particle field
        if functioncalls%nsteps_save_pfield==0 && functioncalls!=1 && save_path!=nothing && save_pfield
            println("Saving PFIELD as: ", run_name*"_pfield.jld\n")
            JLD.save(joinpath(save_path, run_name*"_pfield_$(functioncalls).jld"),
                "PFIELD", pfield)
        end

        return false # breaks out of simulation if true
    end

    return rotorMonitor
end

"Note: assumes that Vinf() is spacially constant"
function getWingMonitor(simulation, nsteps, Vinf;
    figname="monitor", 
    nsteps_plot = 1,
    disp_plot = true, 
    figsize_factor = 1.0,
    save_path = nothing, 
    run_name = "defaultrunname",
    nsteps_savefig = 10,
    extra_plots = true,
    magVinf = nothing,
    rhoinf = 1.225,
    wake_coupled = true
)
    # ------------- SIMULATION MONITOR -----------------------------------------
    wings = extractWings(simulation.vehicle.vlm_system)
    bs = zeros(length(wings))
    y2bs = []
    ars = zeros(length(wings))
    lencrits = zeros(length(wings))

    for (i,wing) in enumerate(wings)
        bs[i] = wing._yn[end] - wing._yn[1]
        y2b = 2*wing._ym/bs[i]
        push!(y2bs, y2b)
        meanchord = mean(getChords(wing))
        ars[i] = bs[i] / meanchord
        lencrits[i] = 0.5*meanchord/vlm.get_m(wing)
    end

    prev_wings = []

    # Name of convergence file
    if save_path!=nothing
        fname = joinpath(save_path, run_name)
    end

    function wingMonitor(sim, PFIELD, T, DT)
        # plot style/colors
        aux = PFIELD.nt/nsteps
        clr = (1-aux, 0, aux)
        # freestream properties
        magVinf = norm(sim.vehicle.V .- Vinf(sim.vehicle.system.O, T))
        qinf = 0.5 * rhoinf * magVinf^2
        if PFIELD.nt==0 && (disp_plot || save_path!=nothing)
            for (wingi, wing) in enumerate(wings)
                figure(figname*"_wing$wingi", figsize=[7*2, 5*2]*figsize_factor)
                subplot(221)
                xlim([-1,1])
                xlabel(L"$\frac{2y}{b}$")
                ylabel(L"$\frac{Cl}{CL}$")
                title("Spanwise lift distribution")

                subplot(222)
                xlim([-1,1])
                xlabel(L"$\frac{2y}{b}$")
                ylabel(L"$\frac{Cd}{CD}$")
                title("Spanwise drag distribution")

                subplot(223)
                xlabel("Simulation time (s)")
                ylabel(L"Lift Coefficient $C_L$")

                subplot(224)
                xlabel("Simulation time (s)")
                ylabel(L"Drag Coefficient $C_D$")

                figure(figname*"_wing$wingi"*"_2", figsize=[7*2, 5*1]*figsize_factor)
                subplot(121)
                xlabel(L"$\frac{2y}{b}$")
                ylabel(L"Circulation $\Gamma$ [m^2/s]")
                subplot(122)
                xlabel(L"$\frac{2y}{b}$")
                ylabel(L"Effective velocity $V_\infty$ [m/s]")

                if extra_plots
                    for num in 1:3 # (1==ApA, 2==AB, 3==BBp)
                        figure(figname*"_wing$wingi"*"_3_$num", figsize=[7*3, 5*3]*figsize_factor)
                        suptitle("Velocitiy at $(num==1 ? "ApA" : num==2 ? "AB" : "BBp")")
                        for i in 7:9; subplot(330+i); xlabel(L"$\frac{2y}{b}$"); end
                        subplot(331)
                        ylabel(L"$V_\mathrm{VPM}$ Velocity [m/s]")
                        subplot(334)
                        ylabel(L"$V_\mathrm{VLM}$ Velocity [m/s]")
                        subplot(337)
                        ylabel(L"$V_\mathrm{kin}$ and $V_\infty$ Velocity [m/s]")
                        subplot(331)
                        title(L"$x$-component")
                        subplot(332)
                        title(L"$y$-component")
                        subplot(333)
                        title(L"$z$-component")
                    end
                    figure(figname*"_wing$wingi"*"_4", figsize=[7*3, 6*1]*figsize_factor)
                    for num in 1:3
                        subplot(130+num)
                        title("Length at $(num==1 ? "ApA" : num==2 ? "AB" : "BBp")")
                        xlabel(L"$\frac{2y}{b}$")
                        if num==1; ylabel("Bound-vortex length"); end;
                    end
                end

                # Convergence file header
                if save_path!=nothing
                    f = open(fname*"_wing$(wingi)_CLCD.csv", "w")
                    print(f, "T,CL,CD\n")
                    close(f)
                end

                push!(prev_wings, deepcopy(wing))
            end
        end


        if PFIELD.nt!=0 && PFIELD.nt%nsteps_plot==(nsteps_plot - 1) && (disp_plot || save_path!=nothing)
            for (wingi, wing) in enumerate(wings)
                prev_wings[wingi] = deepcopy(wing)
            end
        end

        if PFIELD.nt!=0 && PFIELD.nt%nsteps_plot==0 && (disp_plot || save_path!=nothing)
            for (wingi, wing) in enumerate(wings)
                figure(figname*"_wing$wingi")

                # Force at each VLM element
                Ftot = uns.calc_aerodynamicforce(wing, prev_wings[wingi], PFIELD, Vinf, DT,
                                                rhoinf; t=PFIELD.t,
                                                lencrit=lencrits[wingi])
                L, D, S = uns.decompose(Ftot, [0,0,1], [-1,0,0])
                vlm._addsolution(wing, "L", L)
                vlm._addsolution(wing, "D", D)
                vlm._addsolution(wing, "S", S)

                # Force per unit span at each VLM element
                Vout, lenout = extra_plots ? ([], []) : (nothing, nothing)
                ftot = uns.calc_aerodynamicforce(wing, prev_wings[wingi], PFIELD, Vinf, DT,
                            rhoinf; t=PFIELD.t, per_unit_span=true,
                            Vout=Vout, lenout=lenout,
                            lencrit=lencrits[wingi])
                l, d, s = uns.decompose(ftot, [0,0,1], [-1,0,0])

                # Lift of the wing
                Lwing = norm(sum(L))
                CLwing = Lwing/(qinf*bs[wingi]^2/ars[wingi])
                ClCL = norm.(l) / (Lwing/bs[wingi])

                # Drag of the wing
                Dwing = norm(sum(D))
                CDwing = Dwing/(qinf*bs[wingi]^2/ars[wingi])
                CdCD = [sign(dot(this_d, [1,0,0])) for this_d in d].*norm.(d) / (Dwing/bs[wingi]) # Preserves the sign of drag

                vlm._addsolution(wing, "Cl/CL", ClCL)
                vlm._addsolution(wing, "Cd/CD", CdCD)

                subplot(221)
                plot(y2bs[wingi], ClCL, "-", label="FLOWVLM", alpha=0.5, color=clr)

                subplot(222)
                plot(y2bs[wingi], CdCD, "-", label="FLOWVLM", alpha=0.5, color=clr)

                subplot(223)
                plot([T], [CLwing], "o", label="FLOWVLM", alpha=0.5, color=clr)

                subplot(224)
                plot([T], [CDwing], "o", label="FLOWVLM", alpha=0.5, color=clr)

                figure(figname*"_wing$wingi"*"_2")
                subplot(121)
                plot(y2bs[wingi], wing.sol["Gamma"], "-", label="FLOWVLM", alpha=0.5, color=clr)
                if wake_coupled && PFIELD.nt!=0
                    subplot(122)
                    plot(y2bs[wingi], norm.(wing.sol["Vkin"]), "-", label="FLOWVLM", alpha=0.5, color=[clr[1], 1, clr[3]])
                    if "Vvpm" in keys(wing.sol)
                        plot(y2bs[wingi], norm.(wing.sol["Vvpm"]), "-", label="FLOWVLM", alpha=0.5, color=clr)
                    end
                    plot(y2bs[wingi], [norm(Vinf(vlm.getControlPoint(wing, i), T)) for i in 1:vlm.get_m(wing)],
                                                "-k", label="FLOWVLM", alpha=0.5)
                end

                if extra_plots
                    m = vlm.get_m(wing)
                    for num in 1:3               # (1==ApA, 2==AB, 3==BBp)
                        figure(figname*"_wing$wingi"*"_3_$num")
                        for Vi in 1:3           # (1==Vvpm, 2==Vvlm, 3==Vinf && Vkin)
                            for xi in 1:3       # (1==Vx, 2==Vy, 3==Vz)
                                subplot(330 + (Vi-1)*3 + xi)
                                if Vi!=3
                                    plot(y2bs[wingi], [Vout[(i-1)*3 + num][Vi][xi] for i in 1:m], color=clr, alpha=0.5)
                                else
                                    plot(y2bs[wingi], [Vout[(i-1)*3 + num][Vi][xi] for i in 1:m], "k", alpha=0.5)
                                    plot(y2bs[wingi], [Vout[(i-1)*3 + num][Vi+1][xi] for i in 1:m], color=clr, alpha=0.5)
                                end
                            end
                        end
                    end
                    figure(figname*"_wing$wingi"*"_4")
                    for num in 1:3
                        subplot(130+num)
                        plot(y2bs[wingi], [lenout[(i-1)*3 + num] for i in 1:m], color=clr, alpha=0.5)
                    end
                end
                if save_path!=nothing

                    # Write CL and CD values to file
                    f = open(fname*"_wing$(wingi)_CLCD.csv", "a")
                        print(f, T, ",", CLwing, ",", CDwing, "\n")
                    close(f)

                    # write cl, cd, and v values to files
                    f = open(fname*"_wing$(wingi)_clcdv_$(PFIELD.nt).csv", "w")
                        num = 2
                        writedlm(f, ["2y/b" "cl/CL" "cd/CD" "Vvlm_x" "Vvlm_y" "Vvlm_z" "Vvpm_x" "Vvpm_y" "Vvpm_z" "Vkin_x" "Vkin_y" "Vkin_z" "Vinf_x" "Vinf_y" "Vinf_z"],',')
                        writedlm(f, [y2bs[wingi] ClCL CdCD hcat([Vout[(i-1)*3 + num][2] for i in 1:m]...)' hcat(wing.sol["Vvpm"]...)' hcat(wing.sol["Vkin"]...)' hcat([Vinf(vlm.getControlPoint(wing, i), T) for i in 1:vlm.get_m(wing)]...)'], ',')
                    close(f)

                    # Save figures
                    if PFIELD.nt%nsteps_savefig==0
                        figure(figname*"_wing$wingi")
                        savefig(joinpath(save_path, run_name*"_wingconvergence_liftdrag.svg"),
                                                        transparent=false)
                        figure(figname*"_wing$wingi"*"_2")
                        savefig(joinpath(save_path, run_name*"_wingconvergence_circulation.svg"),
                                                        transparent=false)

                        figure(figname*"_wing$wingi"*"_3_2")
                        savefig(joinpath(save_path, run_name*"_wingconvergence_velocity.svg"))
                    end
                end
                prev_wings[wingi] = deepcopy(wing)

            end
        end


        return false
    end
end

function getChords(wing::vlm.Wing)
    dxs2 = (wing._xlwingdcr .- wing._xtwingdcr) .^ 2
    dzs2 = (wing._zlwingdcr .- wing._ztwingdcr) .^ 2
    chords = sqrt.(dxs2 .+ dzs2)
    return chords
end

"Returns a vector of all `::FLOWVLM.Wing` objects contained in the ::`FLOWVLM.WingSystem`"
function extractWings(wingsystem::vlm.WingSystem)
    wings = Array{vlm.Wing, 1}()
    for wing in wingsystem.wings
        extractWings!(wings, wing)
    end
    
    return wings
end

function extractWings!(wingvector::Array{vlm.Wing,1}, addthis::vlm.WingSystem)
    for wing in addthis.wings
        extractWings!(wingvector, wing)
    end
end

function extractWings!(wingvector::Array{vlm.Wing,1}, addthis::vlm.Wing)
    push!(wingvector, addthis)
end

"""
Simple integration method.
"""
function integratetrap(x::Union{Array{T,1}, Array{T,2}}, y::Array{T,1}; scaley = 1.0) where T
    integrand = 0
    for i = 1:length(x) - 1
        integrand += (x[i+1] - [i]) * (y[i+1] + y[i]) / 2 * scaley
    end
    return integrand
end

function integratetrap(x::Union{Array{T,1}, Array{T,2}}, yfunction; scaley = 1.0) where T
    integrand = 0
    for i = 1:length(x) - 1
        # println("Sherlock! integratetrap : UnsTools.jl : 1076 : \n\ti=$i\n\tx[$i]=$(x[i])\n\tyfunction($(x[i])) = $(yfunction(x[i]))")
        integrand += (x[i+1] - x[i]) * (yfunction(x[i+1]) + yfunction(x[i])) / 2 * scaley
    end
    return integrand
end

"""
chopWake!(pfield::vpm.ParticleField, cut::String, location)

Chops the wake according to the directions specified.

If `cut == "plane"`, removes particles at all locations satisfying:

* x_particle[px, py, pz] s.t. a*px + b*py + c*pz > 1, where
* location : [a, b, c] describes the equation for the plane ax + by + cz = 1

E.g., to remove all particles above the plane x = 2.0, run:

```julia
chopWake!(pfield, "plane", [0.5, 0.0, 0.0])
```

If `cut == "cylinder"`, removes particles at all locations that don't lie inside a cylinder defined by:

* location : [radius_rotor1::Number, radius_rotor2::Number...] describes the radii of the cylinders created by any of the rotors (must be length Nrotors)

E.g., to remove all particles outside a cylinder with radius 0.25R_rotor aligned with the system's single rotor

```julia
chopWake!(pfield, "cylinder", [0.25])
```

If `cut == "timeshed"`, removes all particles shed at times within the inclusive bounds:

* location : [lowerboundtimestep, upperboundtimestep]

E.g., to remove all particles shed from steps 5 to 50, run:

```julia
chopWake!(pfield, "timeshed", [5,50])
```
"""
function chopWake!(pfield::vpm.ParticleField, cut::String, location; rotors=[], verbose=true, v_lvl=0, reverselogic=false)
    println("\t"^v_lvl * "Chopping wake...")
    if cut == "plane"
        println("\t"^v_lvl,"Begin plane chop...")
        # println("\t"^v_lvl * "Sherlock! commencing `plane` cut...")
        # println("\t"^v_lvl * "Sherlock! N particles = ",pfield.np)
        a = location[1]
        b = location[2]
        c = location[3]
        if a < 0 || b < 0 || c < 0
            throw("negative parameters a, b, c not supported")
        end
        # println("\t\tpfield._p_field[pfield.np,:] = ",pfield._p_field[pfield.np,:])
        ineq = reverselogic ? "<" : ">"
        println("\t"^v_lvl * "Chopping all px " * ineq * "$(round(1/a,2))")
        numdeleted = 0
        for  pi = pfield.np:-1:1
            x = vpm.get_x(pfield, pi)
            chopaway = (a*x[1] + b*x[2] + c*x[3]) > 1.0
            reverselogic ? chopaway = !chopaway : nothing
            if chopaway
                println("\t"^(v_lvl+1),"Deleting particle n=",pi)
                vpm.delparticle(pfield, pi)
                numdeleted += 1
            end
        end
        println("\t"^v_lvl,"particles deleted: ",numdeleted)
    elseif cut == "cylinder"
        println("\t"^v_lvl,"Begin cylinder chop...")
        numdeleted = 0
        if length(rotors) != length(location); throw("Location vector for cylinder chop inconsistent with number of rotors."); end
        for pi = pfield.np:-1:1
            incylinders = Bool[]
            x = vpm.get_x(pfield, pi)            
            for rotori = 1:length(rotors)
                point = rotors[rotori]._wingsystem.O
                δx = x .- point
                orientation = rotors[rotori]._wingsystem.Oaxis[:,1]
                radius = location[rotori]
                incylinder = norm(δx .- dot(δx, orientation)) < radius * rotors[rotori].rotorR
                reverselogic ? chopaway = !chopaway : nothing
                push!(incylinders,incylinder)
            end
            if !(true in incylinders)
                println("\t"^(v_lvl+1), "Deleting particle n=",pi)
                vpm.delparticle(pfield, pi)
                numdeleted += 1
            end
        end
    elseif cut == "timeshed"
        println("Sherlock! \n\tpfield._timeshed = $(pfield._timeshed)\n")
        chopthese = find(x -> x >= location[1] && x <= location[2], pfield._timeshed)
        for pi = pfield.np:-1:1
            if pi in chopthese
                println("\t"^(v_lvl+1), "Deleting particle n=",pi)
                vpm.delparticle(pfield, pi)
            end
        end
        println("\tpfield._timeshed = $(pfield._timeshed)\n")
    elseif cut == "maxgamma"
        println("Sherlock! removing high circulation particles")
        chopthese = find(x -> x >= location, [maximum(pfield._p_field[i,4:6]) for i in 1:size(pfield._p_field)[1]])
        for pi = pfield.np:-1:1
            if pi in chopthese
                println("\t"^(v_lvl+1), "Deleting particle n=",pi)
                vpm.delparticle(pfield, pi)
            end
        end
    else
        throw("Desired `cut` not found.")
    end
    println("\t"^v_lvl * "Finished chopping wake.")
    println("")
end

"""
test the logic of the `reverselogic` variable in chopWake!()
"""
function testChopWake(x,location,reverselogic)
    a = location[1]
    b = location[2]
    c = location[3]
    if (a*x[1] + b*x[2] + c*x[3]) * (-1)^reverselogic > (-1)^reverselogic
        return true
    else
        return false
    end
end

end # module
