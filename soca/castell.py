import re
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

# NOTA: Modelar primer cordó (+ segones mans)
# 3dX = Rengla [0] (pesada), plena [1] (pesada), buida [2]
# 4dX = Rengles del dos [0, 2] (pesades), rengles de l'acotxador i enxaneta [1, 3]
# 2dX = iguals de pesants (pesades)
# Castells amb suports més grans de 4:
#   - 5dX: 3dX + 2dX
#   - 7dX: 3dX + 2dX + 2dX
#   - 9dX: 3dX + 3dX + 3dX

class Descripcio():
    def __init__(self, castell: str):
        descripcio = self.nomenclatura_a_descripcio(castell)
        self.columnes = descripcio['columnes']
        self.pisos = descripcio['pisos']
        self.agulla = descripcio['agulla']
        self.suports = descripcio['suports']
        self.pisos_suport = descripcio['pisos_suport']

    def nomenclatura_a_descripcio(self, castell: str) -> Dict[str, Any]:
        """
        Convert the nomenclature of a castell to a description.
        """
        patro_nomenclatura = re.compile(
            r'^(?P<columnes>p|[2-9]\d*)'
            r'd'
            r'(?P<pisos>\d+)'
            r'(?P<agulla>a)?'
            r'(?:(?P<suports>n|f(?:m(?:p)?)?))?$'
        )
        if (match := patro_nomenclatura.match(castell)) is not None:
            num_columnes = match.group('columnes')
            if num_columnes == 'p':
                num_columnes = 1
            else:
                num_columnes = int(num_columnes)
            suports = match.group('suports')
            if suports is None:
                suports = ''
        else:
            raise ValueError(f"Invalid nomenclature: {castell}")
        pisos_suport = 1
        if suports == 'n':
            pisos_suport = 0
        else:
            pisos_suport += sum((
                'f' in suports,
                'm' in suports,
                'p' in suports
            ))
        return {
            'columnes': num_columnes,
            'pisos': int(match.group('pisos')),
            'agulla': match.group('agulla') is not None,
            'suports': suports,
            'pisos_suport': pisos_suport
        }


    def descripcio_a_nomenclatura(self) -> str:
        """
        Convert the description of a castell to its nomenclature.
        """
        if self.columnes == 1:
            columnes = 'p'
        else:
            columnes = str(self.columnes)
        agulla = 'a' if self.agulla else ''
        return f"{columnes}d{self.pisos}{agulla}{self.suports}"


DEFAULT_SKILL_SCORES: Dict[str, float] = {
    k: 1.0 for k in [
        "enxaneta", "acotxador",
        "dosos_oberts", "dosos_tancats",
        "tronc_alt_lleuger", "tronc_alt_pesat",       # terços, quarts
        "tronc_baix_lleguer", "tronc_baix_pesat",     # baixos, segons
        "contrafort", "agulla",
        "crossa", "mans",
        "lateral", "daus",
    ]
}

class Expertesa:
    def __init__(self, **scores):
        clip = lambda score: float(max(1.0, min(10.0, score)))
        for k, default in DEFAULT_SKILL_SCORES.items():
            try:
                value = float(scores.get(k, default))
            except (TypeError, ValueError):
                raise TypeError(f"Score for '{k}' is {type(scores[k])}; must be numeric")
            setattr(self, k, clip(value))

class Casteller():
    def __init__(
        self,
        nom: str,
        cognoms: str = None,
        alçada: float = None,
        pes: float = None,
        amplada_peus: float = None,
        amplada_espatlla: float = None,
        expertesa: Expertesa = None
    ):
        self.nom, self.nom = nom, cognoms
        self.nom_complet = f"{nom} {cognoms.upper()}"
        if pes is not None and pes < 0:
            raise ValueError("Pes ha de ser un valor positiu")
        self.pes = pes
        if alçada is not None and alçada < 0:
            raise ValueError("L'alçada ha de ser un valor positiu")
        self.alçada = alçada
        if amplada_peus is not None and amplada_peus < 0:
            raise ValueError("L'amplada de peus ha de ser un valor positiu")
        self.amplada_peus = amplada_peus
        if amplada_espatlla is not None and amplada_espatlla < 0:
            raise ValueError("L'amplada d'espatlla ha de ser un valor positiu")
        self.amplada_espatlla = amplada_espatlla
        self.expertesa = expertesa or Expertesa()


def is_ragged(nested):
    """
    Check if `nested` (list of lists/tuples/etc) is ragged:
    returns True if inner sequences differ in length.
    """
    lengths = [len(inner) for inner in nested]
    return not all(l == lengths[0] for l in lengths)


class Castell():
    def __init__(self, nom: str):
        self.metadata = Descripcio(nom)
        self.metadata
        self.tronc, self.pom, self.suports = self._inicialitzar_components()

    def __str__(self):
        return f"""\
Castell {self.metadata.descripcio_a_nomenclatura()}:

# TODO: Pintar el tronc i la pinya per separat
"""

    def _inicialitzar_components(self):
        # initialize tronc as a numpy array matrix of shape (descripcio.pisos, descripcio.columnes)
        tronc = np.empty((self.metadata.pisos-3, self.metadata.columnes), dtype=object)
        # Pom només té dosos si columnes > 1
        pom = ((np.empty(2, dtype=object),) if self.metadata.columnes > 1 \
            else ()) + (
            np.empty(1, dtype=object),
            np.empty(1, dtype=object)
        )

        suports = dict(
            agulles = np.empty((self.metadata.pisos_suport, self.metadata.columnes), dtype=object),
            contraforts = np.empty((self.metadata.pisos_suport, self.metadata.columnes), dtype=object),
            daus = np.empty((self.metadata.pisos_suport, self.metadata.columnes), dtype=object),
            mans = np.empty((self.metadata.pisos_suport, self.metadata.columnes, 3), dtype=object),         # 3 = num. mans per rengla
            crosses = np.empty((self.metadata.pisos_suport, self.metadata.columnes, 2), dtype=object),      # 2 = left [0], right [1]
            laterals = np.empty((self.metadata.pisos_suport, self.metadata.columnes, 2, 3), dtype=object)   # See lines above
        )
        return tronc, pom, suports


    def assignar_posicio(self, casteller: Casteller, posicio: dict) -> None:
        if posicio['component'] == 'tronc':
            assert len(posicio['coordenades']) == 2 and all(
                c < s for c, s in zip(posicio['coordenades'], self.tronc.shape)
            )
            self.tronc[posicio['coordenades']] = casteller
        elif posicio['component'] == 'suports':
            correct_coordinate_length = {
                'agulles': 2,
                'contraforts': 2,
                'mans': 3,
                'crosses': 3
            }
            if posicio['subcomponent'] not in self.suports.keys():
                raise ValueError(f"Subcomponent {posicio['subcomponent']} no reconegut. Ha de ser {list(correct_coordinate_length.keys())}.")
            assert len(posicio['coordenades']) == correct_coordinate_length[posicio['subcomponent']]\
            and all(
                c < s for c, s in zip(posicio['coordenades'], self.suports[posicio['subcomponent']].shape)
            )
            self.suports[posicio['subcomponent']][posicio['coordenades']] = casteller
        elif posicio['component'] == 'pom':
            assert len(posicio['coordenades']) == 2\
            and posicio['coordenades'][0] < len(self.pom)\
            and posicio['coordenades'][1] < len(self.pom[posicio['coordenades'][0]])
            self.pom[posicio['coordenades'][0]][posicio['coordenades'][1]] = casteller
        else:
            raise ValueError(f"Component '{posicio['component']}' no reconegut. Ha de ser 'tronc', 'pom' o 'suports'.")

    @staticmethod
    def casteller_attr_matrix(castellers, attribute):
        rows = [[getattr(c, attribute) for c in row] for row in castellers]
        if is_ragged(castellers):
            return np.array(rows, dtype=object)
        return np.array(rows)


    def _agregar_pesos(self, pis_inici: int, pis_final: int) -> np.ndarray:
        tronc_castell_pes = self.casteller_attr_matrix(self.tronc, 'pes')
        (p0, p1), columnes = (pis_inici-1, pis_final-2), tronc_castell_pes.shape[1]
        pes_agregat = np.zeros((p1 - p0, columnes))
        for i in range(p1, p0, -1):
            # TODO: Alguna cosa falla al càlcul tenint en compte el pom
            pes_agregat[i, :] = pes_agregat[i + 1, :] + tronc_castell_pes[i, :]
        return pes_agregat


    def _agregar_alçades(self, pis_inici: int, pis_final: int) -> np.ndarray:
        tronc_castell_alçada = self.casteller_attr_matrix(self.tronc, 'alçada')
        (p0, p1), columnes = (pis_inici, pis_final), tronc_castell_alçada.shape[1]
        alçada_agregada = np.zeros((p1 - p0, columnes))
        alçada_agregada[0, :] = tronc_castell_alçada[0, :]
        for i in range(p0, p1):
            alçada_agregada[i, :] = alçada_agregada[i - 1, :] + tronc_castell_alçada[i, :]
        return alçada_agregada


    def agregar_metrica(self, metrica: str, pis_inici: int = None, pis_final: int = None) -> np.array:
        funcions_agregacio = {"pes": self._agregar_pesos, "alçada": self._agregar_alçades}
        if metrica not in (metriques_acceptades := list(funcions_agregacio.keys())):
            raise ValueError(f"Mètrica ha de ser un de {metriques_acceptades}")
        pis_inici = pis_inici or 0
        pis_final = pis_final or self.tronc.shape[0] + 3
        if pis_final < pis_inici:
            raise ValueError(f"Pis d'inici ha de ser estrictament més petit que el pis final.")
        return funcions_agregacio[metrica](pis_inici, pis_final)


def castellers_aleatoris(
    n: int = 1, seed: int = 1234,
    mu_alçada: float = 175, sd_alçada: float = 4,
    mu_pes: float = 70, sd_pes: float = 10,
    mu_amplada_peus: float = 9.5, sd_amplada_peus: float = 1,
    mu_amplada_espatlla: float = 39, sd_amplada_espatlla: float = 2) -> np.ndarray:
    castellers = []
    np.random.seed(seed)
    for _ in range(n):
        castellers.append(
            Casteller(
                nom='Nom del Casteller',
                alçada=np.random.normal(mu_alçada, sd_alçada),
                pes=np.random.normal(mu_pes, sd_pes),
                amplada_peus=np.random.normal(mu_amplada_peus, sd_amplada_peus),
                amplada_espatlla=np.random.normal(mu_amplada_espatlla, sd_amplada_espatlla)
            )
        )
    return np.array(castellers)


def build_columns(columns_config: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert column layout configuration to base heights for each column.
    
    Parameters
    ----------
    columns_config : dict
        Column layout specification where keys are column names and values indicate
        structural type or number of baixos. Examples:
        {"R": 1, "P": 1, "B": 1} or {"R": 1, "P": 2, "B": 1}
    
    Returns
    -------
    dict
        Mapping of column names to base heights in centimeters.
    """
    # Base heights for different column types (in cm)
    # These are typical heights for the lowest position (baix) in each column type
    COLUMN_BASE_HEIGHTS = {
        1: 175,  # Single baix - standard height
        2: 180,  # Two baixos - slightly taller requirements
        3: 185,  # Three baixos - even taller
    }
    
    # Column type modifiers based on structural role
    COLUMN_TYPE_MODIFIERS = {
        "R": 1.00,  # Rengla - standard load-bearing column
        "P": 1.00,  # Plena - standard load-bearing column  
        "B": 0.95,  # Buida - hollow column, slightly reduced height
    }
    
    result = {}
    
    for col_name, col_value in columns_config.items():
        # Determine base height based on configuration value
        if isinstance(col_value, (int, float)):
            # Number of baixos specified
            num_baixos = int(col_value)
            base_height = COLUMN_BASE_HEIGHTS.get(num_baixos, 175)
        else:
            # Fallback to single baix height
            base_height = 175
        
        # Apply column type modifier
        modifier = COLUMN_TYPE_MODIFIERS.get(col_name, 1.0)
        final_height = base_height * modifier
        
        result[col_name] = round(final_height, 1)
    
    return result

def compute_column_tronc_heights(
    all_assignments: Dict[str, Dict[str, Tuple[Any, ...]]], 
    tronc_positions: list,
    castellers: Optional[pd.DataFrame] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate the actual height of tronc positions per column based on assigned castellers.
    
    Parameters
    ----------
    all_assignments : dict
        Mapping of position_name -> {column_name -> tuple(casteller_ids)}
    tronc_positions : list
        Ordered list of tronc position names to include in calculations
    castellers : pd.DataFrame, optional
        DataFrame with casteller data including height information. If None, returns
        placeholder heights.
    
    Returns
    -------
    dict
        Nested mapping {column_name: {position_name: height_cm}} with actual heights
        of assigned castellers.
    """
    result = {}
    
    # Get all column names from the assignments
    column_names = set()
    for position_assignments in all_assignments.values():
        column_names.update(position_assignments.keys())
    
    for column_name in column_names:
        column_heights = {}
        
        for position_name in tronc_positions:
            # Get assignments for this position and column
            position_assignments = all_assignments.get(position_name, {})
            assigned_ids = position_assignments.get(column_name, ())
            
            if not assigned_ids or assigned_ids == ():
                # No assignment - use default height
                column_heights[position_name] = 175.0
                continue
            
            if castellers is None:
                # No casteller data available - use placeholder
                column_heights[position_name] = 175.0
                continue
            
            # Calculate height based on assigned castellers
            total_height = 0.0
            valid_count = 0
            
            for casteller_id in assigned_ids:
                if casteller_id is None:
                    continue
                    
                try:
                    # Try multiple strategies to find the casteller
                    casteller_row = None
                    
                    # Strategy 1: Try by 'id' column if it exists
                    if 'id' in castellers.columns:
                        casteller_row = castellers[castellers['id'] == casteller_id]
                    
                    # Strategy 2: Try by 'Nom complet' (name-based lookup)
                    if (casteller_row is None or len(casteller_row) == 0) and 'Nom complet' in castellers.columns:
                        casteller_row = castellers[castellers['Nom complet'] == casteller_id]
                    
                    # Strategy 3: Try by index
                    if (casteller_row is None or len(casteller_row) == 0):
                        try:
                            casteller_row = castellers.loc[[casteller_id]]
                        except (KeyError, TypeError):
                            pass
                    
                    if casteller_row is not None and len(casteller_row) > 0:
                        # Get height - try different possible column names
                        height = None
                        for height_col in ['Alçada (cm)', 'alçada', 'alcada', 'height', 'altura']:
                            if height_col in casteller_row.columns:
                                height_val = casteller_row[height_col].iloc[0]
                                if pd.notna(height_val) and height_val > 0:
                                    height = float(height_val)
                                    break
                        
                        if height is not None:
                            total_height += height
                            valid_count += 1
                except (KeyError, IndexError, ValueError, TypeError):
                    # Skip if we can't find the casteller or get their height
                    continue
            
            # Use average height if multiple castellers assigned, or total if single
            if valid_count > 0:
                avg_height = total_height / valid_count
                column_heights[position_name] = round(avg_height, 1)
            else:
                # Fallback to default height
                column_heights[position_name] = 175.0
        
        result[column_name] = column_heights
    
    return result