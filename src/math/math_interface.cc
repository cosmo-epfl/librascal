/**
 * file   math_interface.hh
 *
 * @author  Felix Musil <felix.musil@epfl.ch>
 *
 * @date   14 October 2018
 *
 * @brief defines an interface to other math library like cephes to separate the
 *        namespaces
 *
 * Copyright © 2018  Felix Musil, COSMO (EPFL), LAMMM (EPFL)
 *
 * Rascal is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * Rascal is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this software; see the file LICENSE. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include "math_interface.hh"
#include "math/cephes/mconf.h"

namespace rascal {
  namespace math {

    double hyp1f1(const double & a, const double & b, const double & x) {
      return cephes::hyperg(a, b, x);
    }

    double hyp2f1(const double & a, const double & b, const double & c,
                  const double & x) {
      return cephes::hyp2f1(a, b, c, x);
    }

  }  // namespace math
}  // namespace rascal
