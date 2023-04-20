class Utils:
    IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")

    @staticmethod
    def read_from_xyz_file(spath: Path):
        """Read xyz files and return lists of x,y,z coordinates and atoms"""

        X = []
        Y = []
        Z = []
        atoms = []

        with open(str(spath), "r") as f:

            for line in f:
                l = line.split()
                if len(l) == 4:
                    X.append(float(l[1]))
                    Y.append(float(l[2]))
                    Z.append(float(l[3]))
                    atoms.append(str(l[0]))

        X = np.asarray(X)
        Y = np.asarray(Y)
        Z = np.asarray(Z)

        return X, Y, Z, atoms
    
    @staticmethod
    def generate_png(
        spath: Path,
        dpath: Path,
        resolution=320,
        z_relative=False,
        single_channel_images=False,
    ):
        """Generate a .npy matrix starting from lists of x,y,z coordinates"""

        X, Y, Z, atoms = Utils.read_from_xyz_file(spath)

        if z_relative:
            z_max = np.max(Z)
            z_min = np.min(Z)

            path = spath.parent.joinpath("max_min_coordinates.txt")

            x = np.loadtxt(str(path))

            x_max = x[0][0]
            x_min = x[1][0]

            y_max = x[0][1]
            y_min = x[1][1]

            resolution = round(
                4
                * (
                    5
                    + np.max(
                        [np.abs(x_max), np.abs(x_min), np.abs(y_max), np.abs(y_min)]
                    )
                )
            )
        else:
            path = spath.parent.joinpath("max_min_coordinates.txt")

            x = np.loadtxt(str(path))

            x_max = x[0][0]
            x_min = x[1][0]

            y_max = x[0][1]
            y_min = x[1][1]

            z_max = x[0][2]
            z_min = x[1][2]

            resolution = round(
                4
                * (
                    5
                    + np.max(
                        [np.abs(x_max), np.abs(x_min), np.abs(y_max), np.abs(y_min)]
                    )
                )
            )

        C = np.zeros((resolution, resolution))
        O = np.zeros((resolution, resolution))
        H = np.zeros((resolution, resolution))

        z_norm = lambda x: (x - z_min) / (z_max - z_min)

        C_only = True

        for i in range(len(X)):
            if atoms[i] == "C":
                x_coord = int(round(X[i] * 2) + resolution / 2)
                y_coord = int(round(Y[i] * 2) + resolution / 2)
                if C[y_coord, x_coord] < z_norm(Z[i]):
                    C[y_coord, x_coord] = z_norm(Z[i])
            elif atoms[i] == "O":
                C_only = False
                x_coord = int(round(X[i] * 2) + resolution / 2)
                y_coord = int(round(Y[i] * 2) + resolution / 2)
                if O[y_coord, x_coord] < z_norm(Z[i]):
                    O[y_coord, x_coord] = z_norm(Z[i])
            elif atoms[i] == "H":
                C_only = False
                x_coord = int(round(X[i] * 2) + resolution / 2)
                y_coord = int(round(Y[i] * 2) + resolution / 2)
                if H[y_coord, x_coord] < z_norm(Z[i]):
                    H[y_coord, x_coord] = z_norm(Z[i])

        name = spath.stem

        if single_channel_images:
            C = (C * 255.0).astype(np.uint8)
            O = (O * 255.0).astype(np.uint8)
            H = (H * 255.0).astype(np.uint8)

            image_C = Image.fromarray(C)
            Utils.crop_image(image_C, name + "_C.png", dpath)
            image_O = Image.fromarray(O)
            Utils.crop_image(image_O, name + "_O.png", dpath)
            image_H = Image.fromarray(H)
            Utils.crop_image(image_H, name + "_H.png", dpath)

        else:
            if C_only:
                Matrix = C.copy()
            else:
                Matrix = np.stack((C, O, H), axis=2)
            Matrix = (Matrix * 255.0).astype(np.uint8)
            # Matrix = np.flip(Matrix, 0)

            image = Image.fromarray(Matrix)
            Utils.crop_image(image, name + ".png", dpath)