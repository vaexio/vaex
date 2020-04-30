# from from unreleased traitlets
"""License:
# Licensing terms

Traitlets is adapted from enthought.traits, Copyright (c) Enthought, Inc.,
under the terms of the Modified BSD License.

This project is licensed under the terms of the Modified BSD License
(also known as New or Revised or 3-Clause BSD), as follows:

- Copyright (c) 2001-, IPython Development Team

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

Neither the name of the IPython Development Team nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## About the IPython Development Team

The IPython Development Team is the set of all contributors to the IPython project.
This includes all of the IPython subprojects.

The core team that coordinates development on GitHub can be found here:
https://github.com/jupyter/.

## Our Copyright Policy

IPython uses a shared copyright model. Each contributor maintains copyright
over their contributions to IPython. But, it is important to note that these
contributions are typically only changes to the repositories. Thus, the IPython
source code, in its entirety is not the copyright of any single person or
institution.  Instead, it is the collective copyright of the entire IPython
Development Team.  If individual contributors want to maintain a record of what
changes/contributions they have specific copyright on, they should indicate
their copyright in the commit message of the change, when they commit the
change to one of the IPython repositories.

With this in mind, the following banner should be used in any source code file
to indicate the copyright and license terms:

    # Copyright (c) IPython Development Team.
    # Distributed under the terms of the Modified BSD License.
"""

import copy

try:
    from inspect import Signature, Parameter, signature
except ImportError:
    from funcsigs import Signature, Parameter, signature

from traitlets import Undefined


def _get_default(value):
    """Get default argument value, given the trait default value."""
    return Parameter.empty if value == Undefined else value


def signature_has_traits(cls):
    """Return a decorated class with a constructor signature that contain Trait names as kwargs."""
    traits = [
        (name, _get_default(value.default_value))
        for name, value in cls.class_traits().items()
        if not name.startswith('_')
    ]

    # Taking the __init__ signature, as the cls signature is not initialized yet
    old_signature = signature(cls.__init__)
    old_parameter_names = list(old_signature.parameters)

    old_positional_parameters = []
    old_var_positional_parameter = None  # This won't be None if the old signature contains *args
    old_keyword_only_parameters = []
    old_var_keyword_parameter = None  # This won't be None if the old signature contains **kwargs

    for parameter_name in old_signature.parameters:
        # Copy the parameter
        parameter = copy.copy(old_signature.parameters[parameter_name])

        if parameter.kind is Parameter.POSITIONAL_ONLY or parameter.kind is Parameter.POSITIONAL_OR_KEYWORD:
            old_positional_parameters.append(parameter)

        elif parameter.kind is Parameter.VAR_POSITIONAL:
            old_var_positional_parameter = parameter

        elif parameter.kind is Parameter.KEYWORD_ONLY:
            old_keyword_only_parameters.append(parameter)

        elif parameter.kind is Parameter.VAR_KEYWORD:
            old_var_keyword_parameter = parameter

    # Unfortunately, if the old signature does not contain **kwargs, we can't do anything,
    # because it can't accept traits as keyword arguments
    if old_var_keyword_parameter is None:
        raise RuntimeError(
            'The {} constructor does not take **kwargs, which means that the signature can not be expanded with trait names'
            .format(cls)
        )

    new_parameters = []

    # Append the old positional parameters (except `self` which is the first parameter)
    new_parameters += old_positional_parameters[1:]

    # Append *args if the old signature had it
    if old_var_positional_parameter is not None:
        new_parameters.append(old_var_positional_parameter)

    # Append the old keyword only parameters
    new_parameters += old_keyword_only_parameters

    # Append trait names as keyword only parameters in the signature
    new_parameters += [
        Parameter(name, kind=Parameter.KEYWORD_ONLY, default=default)
        for name, default in traits
        if name not in old_parameter_names
    ]

    # Append **kwargs
    new_parameters.append(old_var_keyword_parameter)

    cls.__signature__ = Signature(new_parameters)

    return cls
