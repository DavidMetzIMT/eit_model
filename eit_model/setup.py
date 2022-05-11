from asyncio.log import logger
from dataclasses import dataclass, field
import numpy as np


@dataclass
class EITPattern:
    """define the overall dimensions of the chamber"""

    # _:KW_ONLY
    type: str = "eit_pattern"
    injAmplitude: float = 1
    injType: str = "{ad}"
    injSpecial: str = "[1 2]"
    measType: str = "{ad}"
    measSpecial: str = "[1 2]"
    patternOption: list = field(default_factory=lambda: ["meas_current"])
    patternFunc: str = "Ring patterning"
    GENERATING_FUNCTIONS: list = field(
        default_factory=lambda: ["Ring patterning", "Array patterning", "3D patterning"]
    )
    PATTERNS: list = None


@dataclass
class EITElecLayout:
    # _:KW_ONLY
    type: str = "eit_elec_layout"
    elecNb: int = 16
    elecForm: str = "Circular"
    elecSize: np.ndarray = field(default_factory=lambda: np.array([0.5000, 0]))
    elecPlace: str = "Wall"
    layoutDesign: str = "Ring"
    layoutSize: float = 4
    zContact: float = 0.0100
    reset: int = 0
    ELEC_FORMS: list = field(
        default_factory=lambda: ["Circular", "Rectangular", "Point"]
    )
    LAYOUT_DESIGN: list = field(
        default_factory=lambda: ["Ring", "Array_Grid 0", "Array_Grid 45"]
    )
    ELEC_PLACE: list = field(default_factory=lambda: ["Wall", "Top", "Bottom"])
    ALLOW_ELEC_PLACEMENT: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class EITChamber:
    # _:KW_ONLY
    type: str = "eit_chamber"
    name: str = "NameDesignOfChamber"
    boxSize: np.ndarray = field(default_factory=lambda: np.array([5., 5., 2.],dtype=float))
    femRefinement: float = 0.5000
    form: str = "Cylinder"
    FORMS: list = field(default_factory=lambda: ["Cylinder", "Cubic", "2D_Circ"])
    ALLOW_ELEC_PLACEMENT: np.ndarray = field(default_factory=lambda: np.array([]))
    height_2D: float=0

    def box_limit(self) -> np.ndarray:
        """Return the Chamber limits as ndarray

        limits= [
            [xmin, ymin (, zmin)]
            [xmax, ymax (, zmax)]
        ]

        if the height of the chamber is zero a 2D box limit is returned

        Returns:
            np.ndarray: box limit
        """
        # logger.debug(f"{self.boxSize=}")
        x = self.length / 2
        y = self.width / 2
        z = self.height / 2
        limits = [[-x, -y, -z], [x, y, z]] if z > 0 else [[-x, -y], [x, y]]
        return np.array(limits, dtype=float)

    def set_box_size(self, val: np.ndarray) -> None:
        val = val.flatten()

        if val.size == 1:
            raise TypeError("val of box size should be [sizex, sizey (, sizez)]")
        elif val.size == 2:
            val = np.append(val, 0)
        elif val.size > 3:
            val = val[:3]

        self.boxSize = val

    @property
    def length(self):
        return self.boxSize[0]

    @property
    def width(self):
        return self.boxSize[1]

    @property
    def height(self):
        return self.boxSize[2]


@dataclass
class EITSetup:
    type: str = "eit_setup"
    chamber: EITChamber = EITChamber()
    elec_layout: EITElecLayout = EITElecLayout()
    pattern: EITPattern = EITPattern()

    def __post_init__(self):
        if isinstance(self.chamber, dict):
            self.chamber = EITChamber(**self.chamber)
        if isinstance(self.elec_layout, dict):
            self.elec_layout = EITElecLayout(**self.elec_layout)
        if isinstance(self.pattern, dict):
            self.pattern = EITPattern(**self.pattern)

    def for_FEModel(self) -> dict:

        return {"refinement": self.chamber.femRefinement}


if __name__ == "__main__":
    import glob_utils.file.mat_utils
    import glob_utils.log.log

    glob_utils.log.log.main_log()

    file_path = "E:/Software_dev/Matlab_datasets/20220307_093210_Dataset_name/Dataset_name_infos2py.mat"
    var_dict = glob_utils.file.mat_utils.load_mat(file_path)
    m = glob_utils.file.mat_utils.MatFileStruct()
    struct = m._extract_matfile(var_dict)
    f = struct["setup"]
    setup = EITSetup(**f)
    print(setup.__dict__)
