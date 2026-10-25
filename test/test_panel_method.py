# add parent directory to import modules
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# import modules
import Modules.PanelMethod as pm
import Modules.SweepAnalysis as swan

# ==================== Positive Tests ====================== #
def test_01():
    """
        Tests NACA0012 airfoil at an alpha of 0 degrees.
    """
    case = pm.SingleCase('test/input/naca0012.yaml')
    case.run()

    assert(abs(case.mesh.c_lift - 0) < 1e-12)
    assert(abs(case.mesh.c_drag - 0.0003220548374481212) < 1e-12)
    assert(abs(case.mesh.c_moment - 0) < 1e-12)

def test_02():
    """
        Tests NACA0012 airfoil at an alpha of 5 degrees.
    """
    case = pm.SingleCase('test/input/naca0012.yaml')
    case.config.alpha = 5
    case.config.update()
    case.run()

    assert(abs(case.mesh.c_lift - 0.6020600851595616) < 1e-12)
    assert(abs(case.mesh.c_drag - 0.0004084944128301751) < 1e-12)
    assert(abs(case.mesh.c_moment - 0.006288802365453155) < 1e-12)

def test_03():
    """
        Tests NACA0012 airfoil at an alpha of -5 degrees.
    """
    case = pm.SingleCase('test/input/naca0012.yaml')
    case.config.alpha = -5
    case.config.update()
    case.run()

    assert(abs(case.mesh.c_lift - -0.6020600851595735) < 1e-12)
    assert(abs(case.mesh.c_drag - 0.0004084944128317433) < 1e-12)
    assert(abs(case.mesh.c_moment - -0.006288802365456267) < 1e-12)

def test_04():
    """
        Tests NACA2414 airfoil at an alpha of 0 degrees.
    """
    case = pm.SingleCase('test/input/naca2414.yaml')
    case.run()

    assert(abs(case.mesh.c_lift - 0.2590173317700819) < 1e-12)
    assert(abs(case.mesh.c_drag - 0.00037690259775470503) < 1e-12)
    assert(abs(case.mesh.c_moment - 0.054145316650591586) < 1e-12)

def test_05():
    """
        Tests NACA2414 airfoil at an alpha of 7.5 degrees.
    """
    case = pm.SingleCase('test/input/naca2414.yaml')
    case.config.alpha = 7.5
    case.config.update()
    case.run()

    assert(abs(case.mesh.c_lift - 1.172428713217867) < 1e-12)
    assert(abs(case.mesh.c_drag - -0.0004499493279104261) < 1e-12)
    assert(abs(case.mesh.c_moment - 0.06674692193809645) < 1e-12)

def test_06():
    """
        Tests NACA2414 airfoil at an alpha of -2.5 degrees.
    """
    case = pm.SingleCase('test/input/naca2414.yaml')
    case.config.alpha = -2.5
    case.config.update()
    case.run()

    assert(abs(case.mesh.c_lift - -0.047042448804226394) < 1e-12)
    assert(abs(case.mesh.c_drag - 0.0007155124061095594) < 1e-12)
    assert(abs(case.mesh.c_moment - 0.050115021061922685) < 1e-12)