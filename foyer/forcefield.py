import itertools
import os
import warnings

from intermol.forces import *
from intermol.gromacs.gromacs_parser import GromacsParser, default_gromacs_include_dir

from foyer.orderedset import OrderedSet
from foyer.atomtyper import find_atomtypes


def apply_forcefield(intermol_system, forcefield, debug=True, nrexcl=4):
    """Apply a forcefield to a Topology. """
    if forcefield.lower() in ['opls-aa', 'oplsaa', 'opls']:
        ff = Forcefield('oplsaa')
    else:
        raise ValueError("Unsupported forcefield: '{0}'".format(forcefield))

    bondgraph = prepare_atoms(intermol_system)
    # Nodes are tuples of (atom, moleculetype).
    atoms = [atom for atom, _ in bondgraph.nodes()]
    find_atomtypes(atoms, forcefield, debug=debug)
    ff.resolve_bondingtypes(bondgraph)
    propogate_atomtyping(intermol_system)
    # print(ff.defines)
    enumerate_forcefield_terms(intermol_system, bondgraph, ff, n_excl=nrexcl)
    update_intermol_system_defaults(ff, intermol_system)

    # import ipdb; ipdb.set_trace()
    # intermol_system.gen_pairs(n_excl=4)
    # ff = ff.prune()


def update_intermol_system_defaults(forcefield, intermol_system):
    """Updates the system defaults. """
    intermol_system.nonbonded_function = forcefield.system.nonbonded_function
    intermol_system.combination_rule = forcefield.system.combination_rule
    intermol_system.genpairs = forcefield.system.genpairs
    intermol_system.lj_correction = forcefield.system.lj_correction
    intermol_system.coulomb_correction = forcefield.system.coulomb_correction
    for atom in intermol_system.atoms:
        key = atom.atomtype[0]
        intermol_system.atomtypes[key] = forcefield.atomtypes[key]


def propogate_atomtyping(intermol_system):
    """Copy atomtype and bondingtype info to each atom instance in the system.

    intermol_system:
    """
    for moltype in intermol_system.molecule_types.values():
        first_mol = None
        for molecule in moltype.molecules:
            if first_mol is None:
                first_mol = molecule
                continue
            for typed_atom, untyped_atom in zip(first_mol.atoms, molecule.atoms):
                untyped_atom._atomtype = typed_atom._atomtype
                untyped_atom.bondingtype = typed_atom.bondingtype


def prepare_atoms(intermol_system):
    """Add neighbors and white- and blacklists to each atom.

    Note
    ----
    The use of ordered sets is not strictly necessary but it helps when
    debugging because it shows the order in which rules are added.

    Parameters
    ----------
    atoms : list of Atom objects
        The atoms whose atomtypes you are looking for. Atoms must have a
        property `neighbors` which is a list of other atoms that they are
        bonded to.

    """
    bondgraph = intermol_system.bondgraph_per_moleculetype
    for atom, mol_type in bondgraph.nodes():
        atom.neighbors = [atom for atom, _ in bondgraph.neighbors((atom, mol_type))]
        atom.whitelist = OrderedSet()
        atom.blacklist = OrderedSet()
    return bondgraph


def enumerate_forcefield_terms(intermol_system, bondgraph, forcefield, angles=True,
                               dihedrals=True, impropers=True, n_excl=4):
    """Convert Bonds to ForcefieldBonds and find angles and dihedrals. """
    create_bonds(intermol_system, forcefield)

    if impropers:
        for key in forcefield.defines:
            if key[0:8] == 'improper':
                name = key.replace('Z', 'X')
                name = name.replace('Y', 'X')
                n_atoms_specified = 4
                btypes = name.split('_')[1:]
                line = btypes + ['1'] + forcefield.defines[key].split()
                line = ' '.join(line)
                improper_type = forcefield.process_forcetype(btypes, 'ProperPeriodicDihedralType',
                                                       line, n_atoms_specified,
                                                       forcefield.gromacs_dihedral_types,
                                                       forcefield.canonical_dihedral)
                improper_type.improper = True
                key = tuple(btypes + [improper_type.improper])
                forcefield.add_improper_types(key, improper_type)

    if any([angles, dihedrals, impropers]):
        for node_1 in bondgraph.nodes_iter():
            neighbors_1 = bondgraph.neighbors(node_1)
            if len(neighbors_1) > 1:
                if angles:
                    create_angles(intermol_system, forcefield, node_1, neighbors_1)
                if dihedrals:
                    for node_2 in neighbors_1:
                        if node_2[0].index > node_1[0].index:
                            neighbors_2 = bondgraph.neighbors(node_2)
                            if len(neighbors_2) > 1:
                                create_dihedrals(intermol_system, forcefield, node_1, neighbors_1, node_2, neighbors_2)
                if impropers and len(neighbors_1) >= 3:
                    create_impropers(intermol_system, node_1, neighbors_1, forcefield)
            node_1[1].nrexcl = n_excl
            if intermol_system.genpairs == 'yes' and n_excl < 4:
                create_pairs(bondgraph, node_1, neighbors_1, n_excl=n_excl)


def create_bonds(intermol_system, forcefield):
    """Convert from tuples of (Atom1, Atom2) to ForcefieldBonds. """
    for mol_type in intermol_system.molecule_types.values():
        for molecule in mol_type.molecules:
            for bond in mol_type.bonds:
                atom1 = molecule.atoms[bond.atom1 - 1]
                atom2 = molecule.atoms[bond.atom2 - 1]
                bondingtypes = tuple([atom1.bondingtype, atom2.bondingtype])
                # TODO: Hide lookup logic.
                try:
                    bondtype = forcefield.bondtypes[bondingtypes]
                except KeyError:
                    try:
                        bondtype = forcefield.bondtypes[bondingtypes[::-1]]
                    except KeyError:
                        raise ValueError('No bondtype exists for bondingtypes {0}'.format(bondingtypes))
                bond.forcetype = bondtype #was bond.bondtype?
            break  # Only loop through one of the molecules.


def create_angles(intermol_system, forcefield, node, neighbors):
    """Find all angles around a node. """
    atom2 = node[0]
    mol_type = node[1]
    neighbor_atoms = [atom for atom, _ in neighbors]

    for pair in itertools.combinations(neighbor_atoms, 2):
        atom1 = pair[0]
        atom3 = pair[1]
        bondingtypes = tuple([atom1.bondingtype, atom2.bondingtype, atom3.bondingtype])
        # TODO: Hide lookup logic.
        try:
            angletype = forcefield.angletypes[bondingtypes]
        except KeyError:
            try:
                angletype = forcefield.angletypes[bondingtypes[::-1]]
            except KeyError:
                raise ValueError('No angletype exists for bondingtypes {0}'.format(bondingtypes))
        angle = Angle(atom1.index, atom2.index, atom3.index)
        angle.forcetype = angletype #.angletype

        intermol_system.angletypes[bondingtypes] = angletype
        mol_type.angles.add(angle)


def create_dihedrals(intermol_system, forcefield, node_1, neighbors_1, node_2, neighbors_2):
    """Find all dihedrals around a pair of nodes. """
    # We need to make sure we don't remove the node from the neighbor lists
    # that we will be re-using in the following iterations.
    neighbors_1 = set(neighbors_1) - {node_2}
    neighbors_2.remove(node_1)

    atom2 = node_1[0]
    atom3 = node_2[0]
    mol_type = node_1[1]
    neighbor1_atoms = [atom for atom, _ in neighbors_1]
    neighbor2_atoms = [atom for atom, _ in neighbors_2]
    for pair in itertools.product(neighbor1_atoms, neighbor2_atoms):
        if pair[0] != pair[1]:
            atom1 = pair[0]
            atom4 = pair[1]
            # skip 'SI' 'OS' dihedrals, temp fix for betacristobalite
            if atom1.bondingtype and atom2.bondingtype in ['SI', 'OS']:
                continue
            # add False at end of bonding types
            bondingtypes = [tuple([atom1.bondingtype, atom2.bondingtype, atom3.bondingtype, atom4.bondingtype, False])]
            bondingtypes.append(tuple([atom4.bondingtype, atom3.bondingtype, atom2.bondingtype, atom1.bondingtype, False]))
            bondingtypes.append(tuple([atom1.bondingtype, atom2.bondingtype, atom3.bondingtype, 'X', False]))
            bondingtypes.append(tuple([atom4.bondingtype, atom3.bondingtype, atom2.bondingtype, 'X', False]))
            bondingtypes.append(tuple(['X', atom2.bondingtype, atom3.bondingtype, 'X', False]))
            bondingtypes.append(tuple(['X', atom3.bondingtype, atom2.bondingtype, 'X', False]))
            # TODO: Hide lookup logic.
            dihedraltype = None
            for bondingtype in bondingtypes:
                try:
                    dihedraltype = forcefield.dihedraltypes[bondingtype]
                    break
                except KeyError:
                    continue
            if dihedraltype is None:
                warnings.warn('No dihedraltype exists for bondingtypes {0}'.format(bondingtypes[0]))
                continue
            dihedral = Dihedral(atom1.index, atom2.index, atom3.index, atom4.index)
            # dihedral.dihedraltype = dihedraltype
            dihedral.forcetype = list(dihedraltype)[0]
            intermol_system.dihedraltypes[bondingtypes[0]] = dihedraltype
            mol_type.dihedrals.add(dihedral)


def create_impropers(intermol_system, node_1, neighbors_1, forcefield):
    """Find all impropers around a node. """
    mol_type = node_1[1]
    for triplet in itertools.combinations(neighbors_1, 3):
        atom1 = node_1[0]
        atom2 = triplet[0][0]
        atom3 = triplet[1][0]
        atom4 = triplet[2][0]
        bondingtypes = [tuple([atom1.bondingtype, atom2.bondingtype, atom3.bondingtype, atom4.bondingtype])]
        bondingtypes.append(tuple(['X', atom1.bondingtype, 'X', 'X', True]))
        bondingtypes.append(tuple([atom2.bondingtype, atom1.bondingtype, 'X', 'X', True]))
        bondingtypes.append(tuple([atom3.bondingtype, atom1.bondingtype, 'X', 'X', True]))
        bondingtypes.append(tuple([atom4.bondingtype, atom1.bondingtype, 'X', 'X', True]))
        bondingtypes.append(tuple([atom2.bondingtype, 'X', atom3.bondingtype, atom4.bondingtype, True]))
        impropertype = None
        for bondingtype in bondingtypes:
            try:
                impropertype = forcefield.impropertypes[bondingtype]
                break
            except KeyError:
                continue
        if impropertype is None:
            warnings.warn('No improper dihedraltype exists for bondingtypes {0}'.format(bondingtypes[0]))
            continue
        dihedral = Dihedral(atom1.index, atom2.index, atom3.index, atom4.index)
        dihedral.forcetype = impropertype

        intermol_system.dihedraltypes[bondingtypes[0]] = impropertype
        mol_type.dihedrals.add(dihedral)


def create_pairs(bondgraph, node_1, neighbors_1, n_excl=3):
    """Find all pairs for a node. """
    atom1 = node_1[0]
    mol_type = node_1[1]
    for node_2 in neighbors_1:
        atom2 = node_2[0]
        neighbors_2 = bondgraph.neighbors(node_2)
        if len(neighbors_2) > 1:
            for node_3 in neighbors_2:
                if node_3[0] != atom1:
                    atom3 = node_3[0]
                    neighbors_3 = bondgraph.neighbors(node_3)
                    if len(neighbors_3) > 1:
                        for node_4 in neighbors_3:
                            if node_4[0] != atom2:
                                atom4 = node_4[0]
                                if node_4[0].index > node_1[0].index:
                                    if n_excl <= 3:
                                        gen_pair((atom1, atom4), mol_type)
                    if node_3[0].index > node_1[0].index:
                        if n_excl <= 2:
                            gen_pair((atom1, atom3), mol_type)
        if node_2[0].index > node_1[0].index:
            if n_excl <= 1:
                gen_pair((atom1, atom2), mol_type)


def gen_pair(pair, mol_type):
    non_existing_pair = True
    for mol_pair in mol_type.pair_forces:
        if pair[0].index == mol_pair.atom1 and pair[1].index == mol_pair.atom2:
            non_existing_pair = False
    if non_existing_pair:
        lj_pair = LjCPair(pair[0].index, pair[1].index)
        mol_type.pair_forces.add(lj_pair)


class Forcefield(GromacsParser):
    """A container class for the OPLS forcefield."""

    def __init__(self, forcefield_string):
        """Populate the database using files bundled with GROMACS."""
        ff_file = os.path.join(default_gromacs_include_dir(),
                               '{0}.ff/forcefield.itp'.format(forcefield_string))
        super(Forcefield, self).__init__(ff_file, None)
        self.read()

    def read(self):
        """ """
        self.current_directive = None
        self.if_stack = list()
        self.else_stack = list()

        self.atomtypes = self.system.atomtypes
        self.bondtypes = self.system.bondtypes
        self.angletypes = self.system.angletypes
        self.dihedraltypes = self.system.dihedraltypes
        self.impropertypes = dict()
        self.implicittypes = dict()
        self.pairtypes = dict()
        self.cmaptypes = dict()
        self.nonbondedtypes = dict()

        # Parse the itp file into a set of plain text, intermediate
        # TopMoleculeType objects.
        self.process_file(self.top_filename)

    def resolve_bondingtypes(self, bondgraph):
        """ """
        from simtk.unit import elementary_charge, atom_mass_units, nanometer, kilojoules, mole, radian, degree

        for atom, _ in bondgraph.nodes():
            # Temporary solution: For Adding missing Forcefield Elements
            if atom.atomtype[0] not in self.atomtypes:
                self.bondtypes[('SI', 'CT')] = HarmonicBondType(bondingtype1='SI', bondingtype2='CT', c=False, length=.1850*nanometer, k=167360*kilojoules/(nanometer**2*mole))
                self.angletypes[('SI', 'CT', 'CT')] = HarmonicAngleType(bondingtype1='SI', bondingtype2='CT', bondingtype3='CT', k=254.97296*kilojoules/(mole*radian**2), theta=120*degree)
                self.angletypes[('SI', 'CT', 'HC')] = HarmonicAngleType(bondingtype1='SI', bondingtype2='CT', bondingtype3='HC', k=418.4*kilojoules/(mole*radian**2), theta=109.5*degree)
                self.angletypes[('OH', 'SI', 'CT')] = HarmonicAngleType(bondingtype1='OH', bondingtype2='SI', bondingtype3='CT', k=502.18*kilojoules/(mole*radian**2), theta=100*degree)
                self.angletypes[('OH', 'SI', 'OH')] = HarmonicAngleType(bondingtype1='OH', bondingtype2='SI', bondingtype3='OH', k=502.18*kilojoules/(mole*radian**2), theta=110*degree)
                self.angletypes[('OS', 'SI', 'CT')] = HarmonicAngleType(bondingtype1='OS', bondingtype2='SI', bondingtype3='CT', k=502.18*kilojoules/(mole*radian**2), theta=100*degree)
                if atom.atomtype[0] in ['opls_1155']:
                    atom.bondingtype = 'HO'
                    atom.charge = (0, 0.215*elementary_charge)
                    atom.mass = (0, 1.00800*atom_mass_units)
                    new_atom_type = AtomSigepsType(atom.atomtype[0], atom.bondingtype, 1, atom.mass[0], atom.charge[0], 'A', 0*nanometer, 0*kilojoules/mole)
                    self.system.add_atomtype(new_atom_type)
                elif atom.atomtype[0] in ['opls_1154']:
                    atom.bondingtype = 'OH'
                    atom.charge = (0, -0.645*elementary_charge)
                    atom.mass = (0, 15.99940*atom_mass_units)
                    new_atom_type = AtomSigepsType(atom.atomtype[0], atom.bondingtype, 1, atom.mass[0], atom.charge[0], 'A', .312*nanometer, .71128*kilojoules/mole)
                    self.system.add_atomtype(new_atom_type)
                elif atom.atomtype[0] in ['opls_1156']:
                    atom.bondingtype = 'OS'
                    atom.charge = (0, -0.645*elementary_charge)
                    atom.mass = (0, 15.99940*atom_mass_units)
                    new_atom_type = AtomSigepsType(atom.atomtype[0], atom.bondingtype, 1, atom.mass[0], atom.charge[0], 'A', .312*nanometer, .71128*kilojoules/mole)
                    self.system.add_atomtype(new_atom_type)
                elif atom.atomtype[0] in ['opls_1000', 'opls_1002']:
                    atom.bondingtype = 'SI'
                    atom.charge = (0, .860*elementary_charge)
                    atom.mass = (0, 28.08550*atom_mass_units)
                    new_atom_type = AtomSigepsType(atomtype=atom.atomtype[0], bondtype=atom.bondingtype, atomic_number=14, mass=atom.mass[0], charge=atom.charge[0], ptype='A', sigma=.400*nanometer, epsilon=.4184*kilojoules/mole)
                    self.system.add_atomtype(new_atom_type)
                elif atom.atomtype[0] in ['opls_1004']:
                    atom.bondingtype = 'SI'
                    atom.charge = (0, .745*elementary_charge)
                    atom.mass = (0, 28.08550*atom_mass_units)
                    new_atom_type = AtomSigepsType(atomtype=atom.atomtype[0], bondtype=atom.bondingtype, atomic_number=14, mass=atom.mass[0], charge=atom.charge[0], ptype='A', sigma=.400*nanometer, epsilon=.4184*kilojoules/mole)
                    self.system.add_atomtype(new_atom_type)
                elif atom.atomtype[0] in ['opls_1005']:
                    atom.bondingtype = 'CT'
                    atom.charge = (0, -.120*elementary_charge)
                    atom.mass = (0, 12.01100*atom_mass_units)
                    new_atom_type = AtomSigepsType(atom.atomtype[0], atom.bondingtype, 6, atom.mass[0], atom.charge[0], 'A', .350*nanometer, .276144*kilojoules/mole)
                    self.system.add_atomtype(new_atom_type)
                elif atom.atomtype[0] in ['opls_1001']:
                    atom.bondingtype = 'OS'
                    atom.charge = (0, -.430*elementary_charge)
                    atom.mass = (0, 15.99940*atom_mass_units)
                    new_atom_type = AtomSigepsType(atom.atomtype[0], atom.bondingtype, 8, atom.mass[0], atom.charge[0], 'A', .300*nanometer, .711280*kilojoules/mole)
                    self.system.add_atomtype(new_atom_type)
                # End Temp Solution!
                else:
                    print("Could not find atomtype: '{0}' in forcefield.".format(atom.atomtype[0]))
            else:
                atom.bondingtype = self.atomtypes[atom.atomtype[0]].bondtype
                if not hasattr(atom, 'charge') or not atom.charge:
                    atom.charge = (0, self.atomtypes[atom.atomtype[0]].charge)
                if not hasattr(atom, 'mass') or not atom.mass:
                    atom.mass = (0, self.atomtypes[atom.atomtype[0]].mass)



    def add_improper_types(self, key, improper_type):
        """Create a list of Improper_types"""
        self.impropertypes[key] = improper_type
    # def find_atom_types(self, bondtype):
    #     """If the id is the atom type, return the AtomType object. """
    #     matching_atom_types = []
    #     bondtype = str(bondtype)
    #     for kind, atomtype in self.atomtypes.items():
    #         if bondtype.endswith('*'):
    #             # id is a wildcard ending in *
    #             prefix = bondtype[:-1]
    #
    #             if atomtype.bondtype.startswith(prefix):
    #                 matching_atom_types.append(kind)
    #             elif kind.startswith(prefix):
    #                 matching_atom_types.append(kind)
    #         else:
    #             # id is not a wildcard
    #             if bondtype == atomtype.bondtype:
    #                 matching_atom_types.append(kind)
    #     return matching_atom_types
    #
    # def prune(self):
    #     """Create force field with only information relevant to topology. """
    #
    #     bonds_to_remove = set()
    #     for bond in self.top.ff_bonds:
    #         if bond.kind not in self.bondtypes:
    #             print("Could not find bondtype: '{0}' in forcefield.".format(bond.kind))
    #             bonds_to_remove.add(bond)
    #     self.top._ff_bonds = self.top._ff_bonds - bonds_to_remove
    #
    #     angles_to_remove = set()
    #     for angle in self.top.ff_angles:
    #         if angle.kind not in self.angletypes:
    #             print("Could not find angletype: '{0}' in forcefield.".format(angle.kind))
    #             angles_to_remove.add(angle)
    #     self.top._ff_angles = self.top._ff_angles - angles_to_remove
    #
    #     dihedrals_to_remove = set()
    #     for dihedral in self.top.ff_dihedrals:
    #         if dihedral.kind not in self.dihedraltypes:
    #             print("Could not find dihedraltype: '{0}' in forcefield.".format(dihedral.kind))
    #             dihedrals_to_remove.add(dihedral)
    #     self.top._ff_dihedrals = self.top._ff_dihedrals - dihedrals_to_remove
    #
    #     impropers_to_remove = set()
    #     for improper in self.top.ff_impropers:
    #         if improper.kind == ('O_2', 'C_2', 'OS', 'CT'):
    #             print("Keeping explicitcly defined improper: '{0}'".format(improper.kind))
    #         elif improper.kind not in self.impropertypes:
    #             print("Could not find impropertype: '{0}' in forcefield.".format(improper.kind))
    #             impropers_to_remove.add(improper)
    #     self.top._ff_impropers = self.top._ff_impropers - impropers_to_remove
    #
    #
    #     # ff = Forcefield(self.topology)
    # ff.atomtypes.update(self.atomtypes)
    #
    # retained_types = [atom.atomtype for atom in self.topology.atoms]
    #
    # # Prune the atom types
    # for atomtype in list(ff.atomtypes.keys()):
    #     if atomtype not in retained_types:
    #         del ff.atomtypes[atomtype]
    #
    #
    # # Prune the bond types, resolving wildcards.
    # for (bondtype1, bondtype2), bond_type in self.bondtypes.items():
    #     atom_types1 = ff.find_atom_types(bondtype1)
    #     atom_types2 = ff.find_atom_types(bondtype2)
    #
    #     # For every combination of matching atom kinds, create a bond type.
    #     for (atom_type1, atom_type2) in itertools.product(atom_types1, atom_types2):
    #         pair = (atom_type1, atom_type2)
    #         ff.bondtypes[pair] = bond_type
    #
    # # Prune the angle types, resolving wildcards.
    # for (bondtype1, bondtype2, bondtype3), angle_type in self.angletypes.items():
    #     atom_types1 = ff.find_atom_types(bondtype1)
    #     atom_types2 = ff.find_atom_types(bondtype2)
    #     atom_types3 = ff.find_atom_types(bondtype3)
    #
    #     # For every combination of matching atom kinds, create an angle type.
    #     for (atom_type1, atom_type2, atom_type3) in itertools.product(atom_types1, atom_types2, atom_types3):
    #         triplet = (atom_type1, atom_type2, atom_type3)
    #         ff.angletypes[triplet] = angle_type
    # return ff
