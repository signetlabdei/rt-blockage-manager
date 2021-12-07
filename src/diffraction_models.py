import math
import numpy as np
import src.geometry as geom
from typing import Union

def _itu_proj(z1:float, z2:float, sub1:float, sub2:float) -> float:
    return np.sqrt((z1-z2)**2 + (sub1-sub2)**2)

def _itu_eq8081a(t:float, r:float, Dr:float, Dt:float) -> float:
    if abs(Dr-Dt)<1e-9:
        raise ZeroDivisionError
    return (t*Dr**2-r*Dt**2-Dt*Dr*(t-r))/(Dr**2-Dt**2)

def _itu_dist(z:float, y1:float, y2:float, x1:float, x2:float) -> float:
    return np.sqrt((z)**2+(y1-y2)**2+(x1-x2)**2)

def empirical_itu(screen: geom.Parallelogram3d, seg: geom.Segment, wavelength: float) -> float:
    # Recommendation ITU-R P.526-15 (10/2019): Propagation by diffraction, Sec. 5.2.1.2

    # T, R
    T = seg.start   
    R = seg.end

    # ITU coordinate system
    O = screen.plane.intersection(seg)

    if O is None:  # ray parallel to plane. No diffraction.
        return 0.0

    if R==O or T==O:
        Warning(f"Corner case: obstacle too close to edge.")
        ea = 1/2  # handle corner case as in ITU-R P.526-15 (10/2019), Sec. 4.1
        return -20*np.log10(abs(1.0-ea))

    ux = (screen.adj1-screen.p0).normalize()
    uy = (screen.adj2-screen.p0).normalize()
    uz = ux.cross(uy)

    rot = (ux, uy, uz)  # (rotation): new coord system basis
    transl = geom.Vector.from_array(O.coord)  # (translation): intersection is the origin of the new reference system
    homotransf_matrix = geom.TransfMatrix(rot, transl)  

    # changing coordinate system
    T1 = homotransf_matrix.change_coord_syst(T)
    xt = T1.x
    yt = T1.y
    zt = T1.z

    R1 = homotransf_matrix.change_coord_syst(R)
    xr = R1.x
    yr = R1.y
    zr = R1.z

    P11 = homotransf_matrix.change_coord_syst(screen.p0)
    P21 = homotransf_matrix.change_coord_syst(screen.adj1)
    P12 = homotransf_matrix.change_coord_syst(screen.adj2)
    x1 = P11.x
    x2 = P21.x
    y1 = P11.y
    y2 = P12.y

    # 77a-b
    r1 = _itu_proj(zr, zt, yr, yt)
    r2 = _itu_proj(zr, zt, xr, xt)

    # 78a-h
    Dt11 = _itu_proj(zt, 0, y1, yt)
    Dt12 = _itu_proj(zt, 0, y2, yt)
    Dt21 = _itu_proj(zt, 0, x1, xt)
    Dt22 = _itu_proj(zt, 0, x2, xt)
    Dr11 = _itu_proj(zr, 0, y1, yr)
    Dr12 = _itu_proj(zr, 0, y2, yr)
    Dr21 = _itu_proj(zr, 0, x1, xr)
    Dr22 = _itu_proj(zr, 0, x2, xr)

    # 80    
    if abs(Dr11-Dt11)>1e-9:
        x11 = _itu_eq8081a(xt,xr,Dr11,Dt11)  # 80a
    else:
        x11 = (xt+xr)/2  # 80b
    
    if abs(Dr12-Dt12)>1e-9:
        x12 = _itu_eq8081a(xt,xr,Dr12,Dt12)  # 80a
    else:
        x12 = (xt+xr)/2  # 80b

    if abs(Dr21-Dt21)>1e-9:
        y21 = _itu_eq8081a(yt,yr,Dr21,Dt21)  # 81a
    else:
        y21 = (yt+yr)/2  # 81b
    
    if abs(Dr22-Dt22)>1e-9:
        y22 = _itu_eq8081a(yt,yr,Dr22,Dt22)  # 81a
    else:
        y22 = (yt+yr)/2  # 81b

    # 82
    y11 = y1
    y12 = y2
    x21 = x1
    x22 = x2

    # 79
    D11 = _itu_dist(zr,yr,y11,xr,x11) + _itu_dist(zt,yt,y11,xt,x11)
    D12 = _itu_dist(zr,yr,y12,xr,x12) + _itu_dist(zt,yt,y12,xt,x12)
    D21 = _itu_dist(zr,yr,y21,xr,x21) + _itu_dist(zt,yt,y21,xt,x21)
    D22 = _itu_dist(zr,yr,y22,xr,x22) + _itu_dist(zt,yt,y22,xt,x22)

    ph11 = np.exp(-1j*2*math.pi*D11/wavelength)
    ph12 = np.exp(-1j*2*math.pi*D12/wavelength)
    ph21 = np.exp(-1j*2*math.pi*D21/wavelength)
    ph22 = np.exp(-1j*2*math.pi*D22/wavelength)

    # TODO check if r is correct
    d1 = geom.distance(O,T)
    d2 = geom.distance(O,R)
    r = np.sqrt(wavelength*d1*d2/(d1+d2))  # eq 2
    Ph = np.exp(-1j*2*math.pi*r/wavelength)  # eq 84

    # 73e-h
    phi11 = np.arctan((y1-yr)/zr)-np.arctan((y1-yt)/zt)
    phi12 = np.arctan((y2-yr)/zr)-np.arctan((y2-yt)/zt)
    phi21 = np.arctan((x1-xr)/zr)-np.arctan((x1-xt)/zt)
    phi22 = np.arctan((x2-xr)/zr)-np.arctan((x2-xt)/zt)

    # 76
    nu11 = 2*np.sqrt((Dt11+Dr11-r1)/wavelength)
    nu12 = 2*np.sqrt((Dt12+Dr12-r1)/wavelength)
    nu21 = 2*np.sqrt((Dt21+Dr21-r2)/wavelength)
    nu22 = 2*np.sqrt((Dt22+Dr22-r2)/wavelength)

    # 75
    G11 = np.cos(phi11/2)*(1/2-np.arctan(1.4*nu11)/math.pi)
    G12 = np.cos(phi12/2)*(1/2-np.arctan(1.4*nu12)/math.pi)
    G21 = np.cos(phi21/2)*(1/2-np.arctan(1.4*nu21)/math.pi)
    G22 = np.cos(phi22/2)*(1/2-np.arctan(1.4*nu22)/math.pi)

    # 74
    ea = (np.sign(phi11)*(1/2-ph11/Ph*G11)-np.sign(phi12)*(1/2-ph12/Ph*G12))*\
        (np.sign(phi21)*(1/2-ph21/Ph*G21)-np.sign(phi22)*(1/2-ph22/Ph*G22))

    return -20*np.log10(abs(1.0-ea))  # 85

def atan_diffraction(screen: geom.Parallelogram3d, seg: geom.Segment, wavelength: float) -> float:
    # Nurmela, Vuokko, et al. "Deliverable D1. 4 METIS channel models."
    # Proc. Mobile Wireless Commun. Enablers Inf. Soc.(METIS) (2015): 1.
    # Sec. 6.2, step 7.
    bottom_left = screen.p0
    bottom_right = screen.adj1
    top_left = screen.adj2

    screen_plane = geom.Plane(bottom_left, bottom_right, top_left)
    screen_plane_intersec = screen_plane.intersection(seg)
    if screen_plane_intersec is None:  # ray parallel to plane. No diffraction.
        return 0

    left = geom.Point(bottom_left.x, bottom_left.y, screen_plane_intersec.z)
    right = geom.Point(bottom_right.x, bottom_right.y, screen_plane_intersec.z)
    bottom = geom.Point(screen_plane_intersec.x, screen_plane_intersec.y, bottom_right.z)
    top = geom.Point(screen_plane_intersec.x, screen_plane_intersec.y, top_left.z)

    f_hor = lateral_atan_diffraction(left, right, seg, wavelength)
    f_ver = lateral_atan_diffraction(top, bottom, seg, wavelength)

    return -20 * math.log10(1 - f_hor * f_ver)  # eq. 6.5


def lateral_atan_diffraction(p1: geom.Point, p2: geom.Point, seg: geom.Segment,
                             wavelength: float) -> float:
    # Nurmela, Vuokko, et al. "Deliverable D1. 4 METIS channel models."
    # Proc. Mobile Wireless Commun. Enablers Inf. Soc.(METIS) (2015): 1.
    # Sec. 6.2, step 7.

    start = seg.start
    end = seg.end
    r = seg.length()
    r_line = geom.Line(start, end - start)

    # diffraction on p1 edge
    dist_start_p1 = (start - p1).length()
    dist_end_p1 = (end - p1).length()

    p1_seg_path_diff = round(dist_end_p1 + dist_start_p1 - r, 9)
    f1 = math.atan(math.pi / 2 * math.sqrt(math.pi / wavelength * p1_seg_path_diff)) / math.pi  # eq. 6.6
    
    # diffraction on p2 edge
    dist_start_p2 = (start - p2).length()
    dist_end_p2 = (end - p2).length()
    p2_seg_path_diff = round(dist_end_p2 + dist_start_p2 - r, 9)
    f2 = math.atan(math.pi / 2 * math.sqrt(math.pi / wavelength * p2_seg_path_diff)) / math.pi  # eq. 6.6

    # check obstruction
    p1_proj, _ = geom.project(p1, r_line)  # project screen edge p1 on the ray segment
    p2_proj, _ = geom.project(p2, r_line)  # project screen edge p2 on the ray segment
    
    p1_dir = p1-p1_proj  # compute p1_proj vector from p1_proj (belonging to the ray) to p1
    p2_dir = p2-p2_proj  # compute p2_proj vector from p2_proj (belonging to the ray) to p2
    # if p1_proj and p2_proj have opposite directions, the ray is between p1 and p2 (obstructs=True)
    obstructs = p2_dir.dot(p1_dir) < 0  
    if obstructs:  # NLOS
        # when the link is in NLOS the plus sign apply to both edges
        f = f1 + f2
    else:  # LOS
        # when the link is in LOS, the edge farthest from the link is in the shadow zone (plus sign) and the other in the LOS zone (minus sign)
        f = max(f1, f2) - min(f1, f2)
    return f


def _compute_nu(h: float, d1: float, d2: float, wavelength: float) -> float:
    # Compute the diffraction parameter nu
    # - Recommendation ITU-R P.526-15 (10/2019): Propagation by diffraction, Sec. 4.1
    # - J. Kunisch and J. Pamp, "Ultra-wideband double vertical knife-edge model
    #   for obstruction of a ray by a person," 2008 IEEE International Conference
    #   on Ultra-Wideband, Hannover, Germany, 2008, pp. 17-20,
    #   doi: 10.1109/ICUWB.2008.4653341.
    
    if (abs(h-d1)<1e-9 or abs(h-d2)<1e-9) and abs(h)<1e-9:
        # corner case: edge position coincides with one of the segment vertices
        return 0.0

    nu = h * math.sqrt(2 / wavelength * (1 / d1 + 1 / d2))
    
    return nu


def ske_itu(edge_pos: geom.Point, edge_dir: geom.Vector, seg: geom.Segment, wavelength: float) -> float:
    # Recommendation ITU-R P.526-15 (10/2019): Propagation by diffraction, Sec. 4.1
    start = seg.start
    end = seg.end
    los_dir = end - start

    # project edge position on los
    proj, _ = geom.project(edge_pos, geom.Line(start, los_dir))
    edge_cross = edge_dir.cross(los_dir)  # edge orientation
    if abs(edge_cross.length()) < 1e-9:  # screen is parallel to seg
        return 0.0
    edge_cross = edge_cross.normalize()  # edge orientation

    edge_proj, _ = geom.project(proj,
                                geom.Line(edge_pos, edge_cross))  # project edge position on the infinitely long edge
    # use edge projection on los as reference for h
    h = math.copysign(1, edge_dir.dot(proj - edge_proj)) * \
        (edge_proj - proj).length()
    d1 = (edge_proj - start).length()
    d2 = (edge_proj - end).length()
    nu = _compute_nu(h,d1,d2,wavelength)

    fr = fast_fresnel_integral(nu)
    c = fr.real
    s = fr.imag

    log_arg = math.sqrt((1 - c - s) ** 2 + (c - s) ** 2) / 2
    if log_arg != 0:
        j = -20 * math.log10(log_arg)
    else:
        j = math.inf

    return j


def dke_itu(screen: geom.Parallelogram3d, seg: geom.Segment, wavelength: float, model: str = 'avg') -> float:
    # Recommendation ITU-R P.526-15 (10/2019): Propagation by diffraction, Sec. 5.1
    # NOTE: "For this case the field in the shadow of the screen may be calculated by considering three knife-edges, i.e. the top and the two sides of the screen." Not valid outside the shadow.
    # NOTE: This method can be used to compute the minimum diffraction loss bound and a rough estimate of the average diffraction loss. Should not be used for precise computations.

    # Added to remove the diffraction effect when the obj does not shadow the propagation path (previously added to handle obstacles behind the TX/RX).
    screen_intersec = screen.intersection(seg)
    if screen_intersec is None:  # screen-ray intersection is outside the ray
        raise RuntimeError("The ITU DKE model supports obstacles shadowing the propagation path. No shadowing found.")
        
    bottom_left = screen.p0
    bottom_right = screen.adj1
    top_left = screen.adj2

    # TODO check if directions are consistent
    top_dir = (bottom_left - top_left)
    right_dir = (bottom_left - bottom_right)

    j_top = ske_itu(top_left, top_dir, seg, wavelength)
    j_right = ske_itu(bottom_right, right_dir, seg, wavelength)
    j_bottom = ske_itu(bottom_left, -top_dir, seg, wavelength)
    j_left = ske_itu(bottom_left, -right_dir, seg, wavelength)

    js_db = (j_top, j_right, j_bottom, j_left)
    js_lin = (10 ** (j_db / 20) for j_db in js_db)

    if model == 'avg':
        j_total = -10 * math.log10(sum((1 / J ** 2 for J in js_lin)))
    elif model == 'min':
        j_total = -20 * math.log10(sum((1 / J for J in js_lin)))
    else:
        raise ValueError(
            f"model can only be 'avg' or 'min', instead, {model=}")

    return j_total


def ske_kunisch(edge_pos: geom.Point, edge_dir: geom.Vector, seg: geom.Segment, wavelength: float,
               unit: str = 'lin') -> Union[float, complex]:
    # J. Kunisch and J. Pamp, "Ultra-wideband double vertical knife-edge model
    # for obstruction of a ray by a person," 2008 IEEE International Conference
    # on Ultra-Wideband, Hannover, Germany, 2008, pp. 17-20,
    # doi: 10.1109/ICUWB.2008.4653341.

    # the obstacle is treated as an infinitely long vertical strip
    start = seg.start
    end = seg.end
    los_dir = end - start

    # project edge position on los
    proj, _ = geom.project(edge_pos, geom.Line(start, los_dir))
    edge_cross = edge_dir.cross(los_dir)
    if abs(edge_cross.length()) < 1e-9:  # seg parallel to screen
        return 0.0
    edge_cross = edge_cross.normalize()  # edge orientation
    
    edge_proj, _ = geom.project(proj,
                                geom.Line(edge_pos, edge_cross))  # project edge position on the infinitely long edge

    # use edge projection on los as reference for ha
    ha = math.copysign(1, edge_dir.dot(proj - edge_proj)) * \
         (edge_proj - proj).length()
    dt = (edge_proj - start).length()
    dr = (edge_proj - end).length()
    nu = _compute_nu(ha,dt,dr,wavelength)

    fr = fast_fresnel_integral(nu)
    c = fr.real
    s = fr.imag

    lin_gain = (1 + 1j) / 2 * ((1 / 2 - c) - 1j * (1 / 2 - s))
    if unit == 'db':
        if lin_gain != 0:
            j_total = -20 * math.log10(abs(lin_gain))
        else:
            j_total = math.inf
        return j_total
    else:
        return lin_gain


def dke_kunisch(screen: geom.Parallelogram3d, seg: geom.Segment, wavelength: float) -> float:
    # J. Kunisch and J. Pamp, "Ultra-wideband double vertical knife-edge model
    # for obstruction of a ray by a person," 2008 IEEE International Conference
    # on Ultra-Wideband, Hannover, Germany, 2008, pp. 17-20,
    # doi: 10.1109/ICUWB.2008.4653341.
    bottom_left = screen.p0
    bottom_right = screen.adj1
    top_left = screen.adj2

    # Added to remove the diffraction effect when the obj is behind RX
    screen_plane = geom.Plane(bottom_left, bottom_right, top_left)
    screen_plane_intersec = screen_plane.intersection(seg)
    if screen_plane_intersec is None:  # screen-ray intersection is outside the ray
        return 0

    # TODO check if directions are consistent
    right_dir = (bottom_left - bottom_right)

    right_gain = ske_kunisch(bottom_right, right_dir, seg, wavelength)
    left_gain = ske_kunisch(bottom_left, -right_dir, seg, wavelength)

    lin_gain = right_gain + left_gain
    db_gain = -20 * math.log10(abs(lin_gain))
    return db_gain


ffi_a = np.array([1.595769140, -0.000001702, -6.808568854, -0.000576361,
                  6.920691902, -0.016898657, -3.050485660, -0.075752419,
                  0.850663781, -0.025639041, -0.150230960, 0.034404779])
ffi_b = np.array([-0.000000033, 4.255387524, -0.000092810, -7.780020400,
                  -0.009520895, 5.075161298, -0.138341947, -1.363729124,
                  -0.403349276, 0.702222016, -0.216195929, 0.019547031])
ffi_c = np.array([0.000000000, -0.024933975, 0.000003936, 0.005770956,
                  0.000689892, -0.009497136, 0.011948809, -0.006748873,
                  0.000246420, 0.002102967, -0.001217930, 0.000233939])
ffi_d = np.array([0.199471140, 0.000000023, -0.009351341, 0.000023006,
                  0.004851466, 0.001903218, -0.017122914, 0.029064067,
                  -0.027928955, 0.016497308, -0.005598515, 0.000838386])
ffi_exp = np.arange(0, 12)


def fast_fresnel_integral(nu: float) -> complex:
    # Recommendation ITU-R P.526-15 (10/2019): Propagation by diffraction, Sec. 2.7
    x = 0.5 * math.pi * nu ** 2

    if x < 4:
        Fc = (np.exp(1j * x) * math.sqrt(x / 4) *  # type: ignore
              np.sum((ffi_a - 1j * ffi_b) * (x / 4) ** ffi_exp))
    else:
        Fc = ((1 + 1j) / 2 + np.exp(1j * x) * math.sqrt(4 / x) *  # type: ignore
              np.sum((ffi_c - 1j * ffi_d) * (4 / x) ** ffi_exp))

    Fc = Fc * np.sign(nu)
    return Fc
