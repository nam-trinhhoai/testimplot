#include "RgtLayerProcessUtil.h"

template<typename InputType>
int	bl_indpol(int ir,InputType *yy,int dimx,int type,int dimx2)
{
	int	ii, ij;
	int	ind_y, ij_min, ij_max;

	ij = ir;
	ind_y = ij / dimx;
	ij_min = ind_y * dimx + std::max(0, ij - dimx2);
	ij_max = ind_y * dimx + std::min(ij + dimx2, dimx - 1);

	/* Blindage */
	if(dimx2 > dimx / 2)
		dimx2 = dimx2 / 2;

	/* Test de la valeur non nulle de la colonne courante */
	ii = bl_pointpol(yy, ij, dimx, dimx2);

	/* Rejet trace raide */
	if(ii == RAIDE ) {
		return(RAIDE);
	}
	if(ii == NEGATIF) {
		/* Recherche sur une polarite negative: pente decroissante */
		if(type == -1 ) {
			/* Pointe sur minimum local */
			while( ij < ij_max &&
                  (ii = bl_pointpol(yy, ij, dimx, dimx2)) == NEGATIF ) {
				ij++;
			}
			if(ij < ij_max)
				return(ij - 1);
			else
				return (RAIDE) ;
		} else {
			/* Pointe sur maximum local */
			while( ij > ij_min &&
                  (ii = bl_pointpol(yy, ij, dimx, dimx2)) == NEGATIF ) {
				ij--;
			}
			if(ij > ij_min)
				return(ij);
			else
				return(RAIDE) ;
		}
	} else {
		/* Recherche sur une polarite positive: pente croissante */
		if(type == 1) {
			/* Pointe sur maximum local */
			while( ij < ij_max &&
                  (ii = bl_pointpol(yy, ij, dimx, dimx2)) == POSITIF ) {
				ij++;
			}
			if(ij < ij_max)
				return(ij - 1);
			else
				return (RAIDE) ;
		} else {
			/* Pointe sur minimum local */
			while( ij > ij_min &&
                  (ii = bl_pointpol(yy, ij, dimx, dimx2)) == POSITIF ) {
				ij--;
			}
			if(ij > ij_min )
				return(ij);
			else
				return (RAIDE) ;
		}
	}
}

template<typename InputType>
int bl_pointpol(InputType *a,int  ij,int dimx,int imarge)
{
	/***************************************************************/
	/*   isign   : signe du produit du point precedent et du point */
	/*                                                    suivant  */
	/*   pmoins  : Difference avec le point precedent              */
	/*   pplus   : Difference avec le point suivant                */
	/*   ideb    : Indice de detection de debut d'un plateau       */
	/*   ifin    : Indice de detection de fin   d'un plateau       */
	/*   idebx   : Indice de detection de debut de la ligne        */
	/*   ifinx   : Indice de detection de fin   de la ligne        */
	/***************************************************************/
	int	isign, pmoins, ideb, ifin, idebx, ifinx;
	int	pplus;

	pmoins = a[(ij)] - a[(ij)-1];
	isign = pmoins * (a[(ij)] - a[(ij)+1]);
	idebx = (ij / dimx) * (dimx);
	ifinx = idebx + dimx - 1;

	if (isign < 0 ) {
		/* le point ij est sur une pente */
		if (pmoins > 0)
			return (POSITIF); /* pente positive */
		else
			return (NEGATIF); /* pente negative */
	} else if (isign > 0) {
		/* le point ij est un extrema pur */
		if (pmoins < 0)
			return (NEGATIF); /* extrema positif */
		else
			return (POSITIF); /* extrema negatif */
	} else {
		/* le point ij est sur un plateau */
		/* detection de la fin et du debut du plateau */
		for (ideb = ij;
			(ideb>idebx+1) && (a[ideb]-a[ideb-1]) == 0 && (ij-ideb < imarge);
			ideb--);
		for (ifin = ij;
			(ifin<ifinx-1) && (a[ifin+1]-a[ifin]) == 0 && (ifin- ij < imarge);
			ifin++);
		pmoins = a[(ij)] - a[ideb-1];
		pplus = a[ij] - a[ifin+1];
		isign = pplus * pmoins;
		if (pmoins == 0 || pplus == 0) {
			if (pmoins != 0) {
				if ( pmoins > 0 )
					return (POSITIF); /* le plateau est sur pente positive */
				else
					return (NEGATIF); /* le plateau est sur pente negative */
			} else if (pplus != 0) {
				if ( pplus > 0 )
					return (NEGATIF); /* le plateau est sur pente negative */
				else
					return (POSITIF); /* le plateau est sur pente positive */
			} else
				return (RAIDE); /* le plateau est une trace raide  */
		}
		if (isign < 0) {
			/* le plateau est sur une pente */
				if ( pmoins > 0 )
					return (POSITIF); /* le plateau est sur pente positive */
				else
					return (NEGATIF); /* le plateau est sur pente negative */
		} else {
			/* le plateau est sommet */
			if ( pmoins > 0 ) {
				/* le plateau est un sommet positif */
				if ( (ij) - ideb < ifin - (ij) )
					return (POSITIF);/* ij dans la premiere moitie */
				else
					return (NEGATIF); /* ij dans la deuxieme moitie */
			} else if (pmoins < 0) {
				/* le plateau est sur sommet negatif */
				if ( (ij) - ideb < ifin - (ij) )
					return (NEGATIF); /* ij dans la premiere moitie */
				else
					return (POSITIF);/* ij dans la deuxieme moitie */
			}
		}
	}
}

/*=============================================================================

Fonction: UtFiltreMedianeX

But     : Filtrage Mediane en Xd'une image

Entree  :
Entree  :
	yy	: tableau (idimy ligne de idimx pixels)
	idimx
	idimy	: dimensions
	lwx	: largeur fenetre en X
Sortie  :
Retour  :
=============================================================================*/
template<typename InputType>
void UtFiltreMeanX(InputType *tab1,InputType *tab2,std::size_t nx,std::size_t fx,std::size_t opt) {
	std::size_t i,j,ix,ind;
	double som;
	tab2[0]=0;
	for (ix=0; ix<=fx; ix++) {
		tab2[0] = tab2[0] +  tab1[ix];
	}
	for (ix=1; ix<=fx; ix++)
		tab2[ix]=tab2[ix -1]+tab1[ix+fx];
	/* Partie centrale */
	for (ix=fx+1; ix<nx-fx; ix++)
		tab2[ix]= tab2[ix -1]+ tab1[ ix + fx]- tab1[ix-fx-1];

	/* Conditions finales */
	for (ix=nx-fx; ix<nx; ix++)
		tab2[ix]=tab2[ix-1]-tab1[ix-fx-1];
	/* Renormalisations */
	if(opt != 0) {
		for (ix=0; ix<=fx; ix++) tab2[ix]/=(ix+fx+1);
		for (ix=fx+1; ix<nx-fx; ix++) tab2[ix] /= (2*fx+1);
		for (ix=nx-fx; ix<nx; ix++) tab2[ix]/=(nx-ix+fx);
	}
}

/*=============================================================================

Fonction: UtFiltreMedianeX

But     : Filtrage Mediane en Xd'une image

Entree  :
Entree  :
	yy	: tableau (idimy ligne de idimx pixels)
	idimx
	idimy	: dimensions
	lwx	: largeur fenetre en X
Sortie  :
Retour  :
=============================================================================*/
template<typename InputType>
void UtFiltreMedianeX(InputType *yy, int idimx, int idimy, int lwx) {
	std::vector<InputType> a;
	int i,j;
	int debf;   /* indice de debut de fenetre */
	int medf;   /* indice du median de la fenetre */
	int p;

	int lwxd2 = lwx/2;
	int lwxm1 = lwx -1;

	a.resize(lwx);

	/* Pour chaque ligne j */
	/***********************/
	for(j = 0; j < idimy;++j) {
		/* Pour chaque pixeli fin de fenetre */
		/*************************************/
		for(i = lwxm1, medf = j*idimx + lwxd2, debf = j*idimx; i < idimx;i++, debf++, medf++) {
			for (p = 0; p < lwx; p++) a[p]=yy[debf+p];

			UtFiltreMedianeOrder(a.data(), lwx);
			yy[medf]=a[lwxd2];
		}
	}
}

/*=============================================================================

Fonction: UtFiltreMedianeOrder

But     : Tri d'un tableau

Entree  :
Entree  :
	yy	: tableau (idimy ligne de idimx pixels)
	idimx
	idimy	: dimensions
	lwx	: largeur fenetre en X
Sortie  :
Retour  :
=============================================================================*/
template<typename InputType>
void UtFiltreMedianeOrder(InputType *a, int lw)
{
	bool done = false;
	int i;
	InputType tmp;

	while (done == false) {
		done = true;

		for (i=1; i< lw; i++) {
			if (a[i-1]<a[i]) {
				tmp = a[i-1];
				a[i-1] = a[i];
				a[i] = tmp;
				done = false;
			}
		}
	}
}

template<typename InputType>
bool vectorCompare(std::vector<InputType> a, std::vector<InputType> b) {
	bool result = a.size() == b.size();
	if (result) {
		std::size_t i=0;
		while (result && i<a.size()) {
			result = a[i]==b[i];
			i++;
		}
	}
	return result;
}

