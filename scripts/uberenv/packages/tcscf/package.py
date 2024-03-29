# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *
import warnings

import socket
import os

from os import environ as env
from os.path import join as pjoin


def cmake_cache_entry(name, value, comment=""):
    """Generate a string for a cmake cache variable"""

    return 'set(%s "%s" CACHE PATH "%s")\n\n' % (name, value, comment)


def cmake_cache_list(name, value, comment=""):
    """Generate a list for a cmake cache variable"""

    indent = 5 + len(name)
    join_str = '\n' + ' ' * indent
    return 'set(%s %s CACHE STRING "%s")\n\n' % (name, join_str.join(value), comment)


def cmake_cache_string(name, string, comment=""):
    """Generate a string for a cmake cache variable"""

    return 'set(%s "%s" CACHE STRING "%s")\n\n' % (name, string, comment)


def cmake_cache_option(name, boolean_value, comment=""):
    """Generate a string for a cmake configuration option"""

    value = "ON" if boolean_value else "OFF"
    return 'set(%s %s CACHE BOOL "%s")\n\n' % (name, value, comment)


class Tcscf(CMakePackage, CudaPackage):
    """Transcorellated self consistent field theory."""

    homepage = "TODO: fill in"
    git      = "TODO: fill in"

    version('main', branch='main', submodules=True)

    depends_on('camp build_type=Release')
    depends_on('camp +cuda', when='+cuda')

    #
    # Virtual packages
    #
    depends_on('blas')
    depends_on('lapack')

    depends_on('boost')

    depends_on('gsl')

    depends_on('raja@0.14.0 ~examples ~exercises build_type=Release')

    # At the moment Umpire doesn't support shared when building with CUDA.
    depends_on('umpire ~examples build_type=Release')

    depends_on('chai +raja +openmp ~examples build_type=Release')

    depends_on('adiak ~mpi build_type=Release')

    depends_on('caliper@2.7.0 +adiak ~mpi ~libunwind ~libdw ~libpfm ~gotcha ~sampler build_type=Release')

    depends_on('qmcpack ~mpi')

    with when('+cuda'):
        for sm_ in CudaPackage.cuda_arch_values:
            depends_on('raja +cuda cuda_arch={0}'.format(sm_), when='cuda_arch={0}'.format(sm_))
            depends_on('umpire +cuda ~shared cuda_arch={0}'.format(sm_), when='cuda_arch={0}'.format(sm_))
            depends_on('chai +raja +cuda cuda_arch={0}'.format(sm_), when='cuda_arch={0}'.format(sm_))
            
            # There's an issue linking to CUPTI when using caliper+cuda
            depends_on('caliper +cuda cuda_arch={0}'.format(sm_), when='cuda_arch={0}'.format(sm_))

    phases = ['hostconfig', 'cmake', 'build', 'install']

    @run_after('build')
    @on_package_attributes(run_tests=True)
    def check(self):
        """Searches the CMake-generated Makefile for the target ``test``
        and runs it if found.
        """
        with working_dir(self.build_directory):
            ctest('-V', '--force-new-ctest-process', '-j 1')

    @run_after('build')
    def build_docs(self):
        if '+docs' in self.spec:
            with working_dir(self.build_directory):
                make('docs')

    def _get_sys_type(self, spec):
        sys_type = str(spec.architecture)
        # if on llnl systems, we can use the SYS_TYPE
        if "SYS_TYPE" in env:
            sys_type = env["SYS_TYPE"]
        return sys_type

    def _get_host_config_path(self, spec):
        var = ''
        if '+cuda' in spec:
            var = '-'.join([var, 'cuda'])

        hostname = socket.gethostname().rstrip('1234567890')
        host_config_path = "%s-%s-%s%s.cmake" % (hostname,
                                                 self._get_sys_type(spec),
                                                 spec.compiler, var)

        dest_dir = self.stage.source_path
        host_config_path = os.path.abspath(pjoin(dest_dir, host_config_path))
        return host_config_path

    def hostconfig(self, spec, prefix, py_site_pkgs_dir=None):
        """
        This method creates a 'host-config' file that specifies
        all of the options used to configure and build Umpire.
        For more details about 'host-config' files see:
            http://software.llnl.gov/conduit/building.html
        Note:
          The `py_site_pkgs_dir` arg exists to allow a package that
          subclasses this package provide a specific site packages
          dir when calling this function. `py_site_pkgs_dir` should
          be an absolute path or `None`.
          This is necessary because the spack `site_packages_dir`
          var will not exist in the base class. For more details
          on this issue see: https://github.com/spack/spack/issues/6261
        """

        #######################
        # Compiler Info
        #######################
        c_compiler = env["SPACK_CC"]
        cpp_compiler = env["SPACK_CXX"]

        #######################################################################
        # By directly fetching the names of the actual compilers we appear
        # to doing something evil here, but this is necessary to create a
        # 'host config' file that works outside of the spack install env.
        #######################################################################

        sys_type = self._get_sys_type(spec)

        ##############################################
        # Find and record what CMake is used
        ##############################################

        cmake_exe = spec['cmake'].command.path
        cmake_exe = os.path.realpath(cmake_exe)

        host_config_path = self._get_host_config_path(spec)
        with open(host_config_path, "w") as cfg:
            cfg.write("#{0}\n".format("#" * 80))
            cfg.write("# Generated host-config - Edit at own risk!\n")
            cfg.write("#{0}\n".format("#" * 80))

            cfg.write("#{0}\n".format("-" * 80))
            cfg.write("# SYS_TYPE: {0}\n".format(sys_type))
            cfg.write("# Compiler Spec: {0}\n".format(spec.compiler))
            cfg.write("# CMake executable path: %s\n" % cmake_exe)
            cfg.write("#{0}\n\n".format("-" * 80))

            #######################
            # Compiler Settings
            #######################

            cfg.write("#{0}\n".format("-" * 80))
            cfg.write("# Compilers\n")
            cfg.write("#{0}\n\n".format("-" * 80))
            cfg.write(cmake_cache_entry("CMAKE_C_COMPILER", c_compiler))
            cfg.write(cmake_cache_entry("CMAKE_CXX_COMPILER", cpp_compiler))

            # use global spack compiler flags
            cflags = ' '.join(spec.compiler_flags['cflags'])
            cxxflags = ' '.join(spec.compiler_flags['cxxflags'])

            if "%intel" in spec:
                cflags += ' -qoverride-limits'
                cxxflags += ' -qoverride-limits'

            if cflags:
                cfg.write(cmake_cache_entry("CMAKE_C_FLAGS", cflags))

            if cxxflags:
                cfg.write(cmake_cache_entry("CMAKE_CXX_FLAGS", cxxflags))

            release_flags = "-O3 -DNDEBUG"
            cfg.write(cmake_cache_string("CMAKE_CXX_FLAGS_RELEASE",
                                        release_flags))
            reldebinf_flags = "-O3 -g -DNDEBUG"
            cfg.write(cmake_cache_string("CMAKE_CXX_FLAGS_RELWITHDEBINFO",
                                        reldebinf_flags))
            debug_flags = "-O0 -g"
            cfg.write(cmake_cache_string("CMAKE_CXX_FLAGS_DEBUG", debug_flags))

            if "%clang arch=linux-rhel7-ppc64le" in spec:
                cfg.write(cmake_cache_entry("CMAKE_EXE_LINKER_FLAGS", "-Wl,--no-toc-optimize"))

            cfg.write("#{0}\n".format("-" * 80))
            cfg.write("# Cuda\n")
            cfg.write("#{0}\n\n".format("-" * 80))

            if "+cuda" in spec:
                cfg.write(cmake_cache_option("ENABLE_CUDA", True))
                cfg.write(cmake_cache_entry("CMAKE_CUDA_STANDARD", 14))

                cudatoolkitdir = spec['cuda'].prefix
                cfg.write(cmake_cache_entry("CUDA_TOOLKIT_ROOT_DIR",
                                            cudatoolkitdir))
                cudacompiler = "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc"
                cfg.write(cmake_cache_entry("CMAKE_CUDA_COMPILER", cudacompiler))

                cmake_cuda_flags = ('-restrict --expt-extended-lambda -Werror '
                                    'cross-execution-space-call,reorder,'
                                    'deprecated-declarations')

                archSpecifiers = ("-mtune", "-mcpu", "-march", "-qtune", "-qarch")
                for archSpecifier in archSpecifiers:
                    for compilerArg in spec.compiler_flags['cxxflags']:
                        if compilerArg.startswith(archSpecifier):
                            cmake_cuda_flags += ' -Xcompiler ' + compilerArg

                if not spec.satisfies('cuda_arch=none'):
                    cuda_arch = spec.variants['cuda_arch'].value
                    cmake_cuda_flags += ' -arch sm_{0}'.format(cuda_arch[0])

                cfg.write(cmake_cache_string("CMAKE_CUDA_FLAGS", cmake_cuda_flags))

                cfg.write(cmake_cache_string("CMAKE_CUDA_FLAGS_RELEASE",
                                            "-O3 -Xcompiler -O3 -DNDEBUG"))
                cfg.write(cmake_cache_string("CMAKE_CUDA_FLAGS_RELWITHDEBINFO",
                                            "-O3 -g -lineinfo -Xcompiler -O3"))
                cfg.write(cmake_cache_string("CMAKE_CUDA_FLAGS_DEBUG",
                                            "-O0 -Xcompiler -O0 -g -G"))

            else:
                cfg.write(cmake_cache_option("ENABLE_CUDA", False))

            cfg.write('#{0}\n'.format('-' * 80))
            cfg.write('# System Math Libraries\n')
            cfg.write('#{0}\n\n'.format('-' * 80))

            cfg.write(cmake_cache_option('ENABLE_LAPACK', True))
            cfg.write(cmake_cache_list('BLAS_LIBRARIES', spec['blas'].libs))
            cfg.write(cmake_cache_list('LAPACK_LIBRARIES', spec['lapack'].libs))

            cfg.write("#{0}\n".format("-" * 80))
            cfg.write("# CAMP\n")
            cfg.write("#{0}\n\n".format("-" * 80))

            cfg.write(cmake_cache_entry("CAMP_DIR", spec['camp'].prefix))

            cfg.write("#{0}\n".format("-" * 80))
            cfg.write("# RAJA\n")
            cfg.write("#{0}\n\n".format("-" * 80))

            cfg.write(cmake_cache_entry("RAJA_DIR", spec['raja'].prefix))

            cfg.write("#{0}\n".format("-" * 80))
            cfg.write("# Umpire\n")
            cfg.write("#{0}\n\n".format("-" * 80))

            cfg.write(cmake_cache_option("ENABLE_UMPIRE", True))
            cfg.write(cmake_cache_entry("UMPIRE_DIR", spec['umpire'].prefix))

            cfg.write("#{0}\n".format("-" * 80))
            cfg.write("# CHAI\n")
            cfg.write("#{0}\n\n".format("-" * 80))

            cfg.write(cmake_cache_option("ENABLE_CHAI", True))
            cfg.write(cmake_cache_entry("CHAI_DIR", spec['chai'].prefix))

            cfg.write("#{0}\n".format("-" * 80))
            cfg.write("# Adiak\n")
            cfg.write("#{0}\n\n".format("-" * 80))

            cfg.write(cmake_cache_option("ENABLE_ADIAK", True))
            cfg.write(cmake_cache_entry("ADIAK_DIR", spec['adiak'].prefix))

            cfg.write("#{0}\n".format("-" * 80))
            cfg.write("# Caliper\n")
            cfg.write("#{0}\n\n".format("-" * 80))

            cfg.write(cmake_cache_option("ENABLE_CALIPER", True))
            cfg.write(cmake_cache_entry("CALIPER_DIR", spec['caliper'].prefix))

            cfg.write("#{0}\n".format("-" * 80))
            cfg.write("# addr2line\n")
            cfg.write("#{0}\n\n".format("-" * 80))

            cfg.write(cmake_cache_option('ENABLE_ADDR2LINE', '+addr2line' in spec))

            cfg.write("#{0}\n".format("-" * 80))
            cfg.write("# Boost\n")
            cfg.write("#{0}\n\n".format("-" * 80))

            cfg.write(cmake_cache_option("ENABLE_BOOST", True))
            cfg.write(cmake_cache_entry("BOOST_DIR", spec['boost'].prefix))

            cfg.write("#{0}\n".format("-" * 80))
            cfg.write("# GSL\n")
            cfg.write("#{0}\n\n".format("-" * 80))

            cfg.write(cmake_cache_option("ENABLE_GSL", True))
            cfg.write(cmake_cache_entry("GSL_DIR", spec['gsl'].prefix))

            cfg.write("#{0}\n".format("-" * 80))
            cfg.write("# QMCPACK\n")
            cfg.write("#{0}\n\n".format("-" * 80))

            cfg.write(cmake_cache_option("ENABLE_QMCPACK", True))
            cfg.write(cmake_cache_entry("QMCPACK_DIR", spec['qmcpack'].prefix))

    def cmake_args(self):
        spec = self.spec
        host_config_path = self._get_host_config_path(spec)

        options = []
        options.extend(['-C', host_config_path])

        # Shared libs
        options.append(self.define_from_variant('BUILD_SHARED_LIBS', 'shared'))

        return options
