import sys
import cv2
import numpy as np
import argparse
import glob
from tqdm import tqdm
try:
    from xmlPAGE import pageData
    import polyapprox as pa
except:
    from baselines.utils.xmlPAGE import pageData
    import baselines.utils.polyapprox as pa


def basic_baseline(Oimg, Lpoly, args, orientation=None):
    """
    """
    # --- Oimg = image to find the line
    # --- Lpoly polygon where the line is expected to be
    try:
        minX = Lpoly[:, 0].min()
        maxX = Lpoly[:, 0].max()
        minY = Lpoly[:, 1].min()
        maxY = Lpoly[:, 1].max()
    except Exception as e:
        print(e)
        return (False, None)
    if orientation is not None and orientation == "v": 
        hor=1
    elif (maxX-minX > (maxY-minY)*args.th_vert):
        #---it is an horizontal line
        hor=0
    else:
        #--- it is a vertical line
        hor=1
    mask = np.zeros(Oimg.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, Lpoly, (255, 255, 255))
    res = cv2.bitwise_and(Oimg, mask)
    bRes = Oimg[minY:maxY, minX:maxX]
    bMsk = mask[minY:maxY, minX:maxX]
    try:
        bRes = cv2.cvtColor(bRes, cv2.COLOR_RGB2GRAY)
        _, bImg = cv2.threshold(bRes, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, cols = bImg.shape
    except:
        return(False, None)
    # --- remove black halo around the image
    bImg[bMsk[:, :, 0] == 0] = 255
    #Cs = np.cumsum(abs(bImg - 255), axis=0)
    Cs = np.cumsum(abs(bImg - 255), axis=hor)
    if hor:
        Cs = Cs.T
        cols, _ = bImg.shape
    maxPoints = np.argmax(Cs, axis=0)
    Lmsk = np.zeros(bImg.shape)
    points = np.zeros((cols, 2), dtype="int")
    # --- gen a 2D list of points
    for i, j in enumerate(maxPoints):
    	points[i, :] = [i, j]
    # --- remove points at post 0, those are very probable to be blank columns
    points2D = points[points[:, 1] > 0]
    if points2D.size <= 5:
    # --- there is no real line
    	return (False, [[0, 0]])
    if args.approx_alg == "optimal":
    # --- take only 100 points to build the baseline
    	if points2D.shape[0] > args.max_vertex:
        	points2D = points2D[
        	np.linspace(
            	0, points2D.shape[0] - 1, args.max_vertex, dtype=int
        	)
        	]
    	(approxError, approxLin) = pa.poly_approx(
        	points2D, args.num_segments, pa.one_axis_delta
    	)
    elif args.approx_alg == "trace":
    	approxLin = pa.norm_trace(points2D, args.num_segments)
    else:
    	approxLin = points2D
    if hor == 1:
        #--- flip points
        approxLin = approxLin[:,[1,0]]
    approxLin[:, 0] = approxLin[:, 0] + minX
    approxLin[:, 1] = approxLin[:, 1] + minY
    return (True, approxLin)

def naive(Oimg, Lpoly, args, id, orientation=None):
    height, width, channels = Oimg.shape
    max_height = height * args.ver_only

    min_x = np.min(Lpoly[:,0])
    max_x = np.max(Lpoly[:,0])
    min_y = np.min(Lpoly[:,1])
    max_y = np.max(Lpoly[:,1])
    
    largo = float(max_x - min_x) / 2.0
    ancho = float(max_y - min_y) / 2.0
    try:
        relacion = ancho/largo
    except Exception as e:
        print(Lpoly, id)
        raise e
    # if id == "Albatross_vol030of055-141-0_130":
    #     print(ancho >=args.min_pix_altura, ancho)
    if orientation is not None and orientation == "v": 
        approxLin = np.array([[max_x, max_y], [max_x, min_y]]) # giramos
    elif relacion > args.th_vert and ancho >=args.min_pix_altura and max_y <= max_height: # giramos
        # print(max_height, max_y, relacion, largo, ancho)
        approxLin = np.array([[max_x, max_y], [max_x, min_y]])
        # print(id)
    else:
        approxLin = np.array([[min_x, max_y], [max_x, max_y]])
    return (True, approxLin)


def get_main():
    ALGORITHMS = {"basic":basic_baseline,
                  "naive": naive}
    parser = argparse.ArgumentParser(
        description="This script support some basic baseline related extractions"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=None,
        help="path to images files directory.",
    )
    parser.add_argument(
        "--page_dir",
        type=str,
        default=None,
        help="path to page-xml files directory.",
    )
    parser.add_argument(
        "--page_ext",
        type=str,
        default="xml",
        help='Extensiong of PAGE-XML files. Default "xml"',
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="naive",
        help="Algorithm to be used to gen the baselines. NOT USED ANYMORE",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Path to save new generated xml files",
    )
    parser.add_argument(
        "--use_orientation",
        type=bool,
        default=True,
        help="Path to save new generated xml files",
    )
    parser.add_argument(
        "--must_line",
        type=str,
        nargs="+",
        default="",
        help="Search for baselines on this regions even if no TextLine found",
    )
    parser.add_argument(
        "--width_alg",
        type=float,
        default=0.1,
        help="Minium width of the line to use the optimal algorithm.",
    )

    parser.add_argument(
        "--num_segments",
        type=int,
        default=4,
        help="",
    )

    parser.add_argument(
        "--max_vertex",
        type=int,
        default=30,
        help="",
    )

    parser.add_argument(
        "--th_vert",
        type=int,
        default=2,
        help="",
    )

    parser.add_argument(
        "--ver_only",
        type=float,
        default=0.15,
        help="",
    )

    parser.add_argument(
        "--min_pix_altura",
        type=int,
        default=20,
        help="",
    )

    parser.add_argument(
        "--approx_alg",
        type=str,
        nargs="+",
        default="optimal",
        help="d",
    )

    args = parser.parse_args()
    # args.num_segments = 4
    # args.max_vertex = 30
    # # args.th_vert = 2
    # args.th_vert = 1
    # args.ver_only = 0.15 #porcent
    # args.min_pix_altura = 20
    
    # args.approx_alg = "optimal"
    get_baselines = ALGORITHMS[args.algorithm]
    return get_baselines, args

def get_relative_width(coords, width):
    min_x = np.min(coords[:,0])
    max_x = np.max(coords[:,0])
    largo = float(max_x - min_x) / 2.0
    rel_width = largo / width
    return rel_width

def main():
    _, args = get_main()
    files = glob.glob(args.page_dir+"/*."+args.page_ext)
    optimal_alg_count = 0
    for page in tqdm(files):
        print(page)
        data = pageData(page)
        data.parse()
        file_name = data.get_image_path()
        if file_name is None:
            file_name = args.img_dir + '/' + data.get_image_name()
        img = cv2.imread(file_name)
        height, width, channels = img.shape
        lines = data.get_region("TextLine")
        if lines is None:
            print("INFO: No TextLines found on page {}".format(page))
            data.save_xml(f_path=args.out_dir + '/'+ data.full_name)
            continue
        for line in lines:
            coords = data.get_coords(line)
            if len(coords) < 3:
                print(f'id {data.get_id(line)} not processed due to the number of coords')
                continue
            orientation = None
            if args.use_orientation:
                orientation = data.get_attrib(line, attrib="orientation")
            relative_width = get_relative_width(coords, width)
            if relative_width >= args.width_alg:
                (valid, baseline) = basic_baseline(img, coords, args, orientation)
                optimal_alg_count += 1
            else:
                (valid, baseline) = naive(img, coords, args, data.get_id(line), orientation)
                
            # exit()
            if valid == True:
                data.add_baseline(pa.points_to_str(baseline), line)
            else:
                print("No baseline found for line {}".format(data.get_id(line)))
        #--- check for TextRegion without TextLine
        regions = data.get_region("TextRegion")
        idx = 0
        for region in regions:
            r_type = data.get_region_type(region)
            if r_type is not None and r_type in args.must_line:
                lines = data.get_childs("TextLine", parent=region)
                if lines == None:
                    coords = data.get_coords(region)
                    if len(coords) < 3:
                        print(f'id {data.get_id(line)} not processed due to the number of coords')
                        continue
                    relative_width = get_relative_width(coords, width)
                    orientation = None
                    if args.use_orientation:
                        orientation = data.get_attrib(line, attrib="orientation")
                    if relative_width >= args.width_alg:
                        (valid, baseline) = basic_baseline(img, coords, args, orientation)
                        optimal_alg_count += 1
                    else:
                        (valid, baseline) = naive(img, coords, args, data.get_id(line), orientation)
                    if valid == True:
                        line = data.add_element("TextLine", "TextLine_extra" + str(idx), r_type, pa.points_to_str(coords), parent=region)
                        idx += 1
                        data.add_baseline(pa.points_to_str(baseline), line)
        data.save_xml(f_path=args.out_dir + '/'+ data.full_name)
    print(f'{optimal_alg_count} times used the optimal alg ')
    







if __name__== "__main__":
    main()