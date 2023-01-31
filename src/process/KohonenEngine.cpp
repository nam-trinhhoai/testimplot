#include "KohonenEngine.h"

#include <QDebug>

#include <cmath>

KohonenEngine::KohonenEngine(long DimInput, long ne_tailleCarte) {
	m_exampleSize = DimInput;
	m_tmapSize = ne_tailleCarte;

	m_tmap.resize(m_exampleSize * m_tmapSize);
}

KohonenEngine::~KohonenEngine() {

}

bool KohonenEngine::trainTmap(float *exemples, long NbExempl, long ne_nbCycles, long ne_tailleVoisin,
			float sigmamax, float sigmamin, float epsilmax,float epsilmin, long voisinag,
			long typenorm) {
	float   passigma, pasepsil, sigma, epsilon, epsilov;
	float   a0 = 5.2;
	float   sigma0;
	unsigned long   is = 8191;
	long    i, j, k, ij, ex, cy, compteur, tailltot, produit;
	long    nbdimcar = 1, nbcycles = 1000;
	long    dimcarte[5];
	std::vector<float> CarteTop_N;
	std::vector<long> indexexe;
	long     ixi, iyi, ixA, iyA, icoul, icoulc = 150;
	long     id1, id2, inew, itailx, itaily;
	long     iix, iiy, itaix, itaiy;
	long     n_ok = 0, ii;
	std::vector<long> hist, flag;
	long     cysup = 1, NN;
	long     compteurV, exV, compteur_c, compteur_e, ic;
	long     ex_opt, ic_opt, ex1;
	long     ier;
	long     nbcarte, nbcartex, ishx, ishy;
	long     flag_passage = 0;
	long     ik,csup,ind,NormeCarteTop;
	double travail, distance, distance_E,w ;
	static float rand_max, rand_min;
	rand_min = 0;
	rand_max = RAND_MAX;
	/*   liste des bloc3Dxxxx.raw  */
	long np=0 ;
	long size_file ;
	long ix,iy,iz;
	long ind_ex ;
	a0 = ne_tailleVoisin;
	nbcycles = ne_nbCycles;

	tailltot = m_tmapSize * m_exampleSize;
	CarteTop_N.resize(tailltot);

	std::vector<float> exempMax, exempMin;
	exempMin.resize(m_exampleSize);
	exempMax.resize(m_exampleSize);

	long DimInput = m_exampleSize;
	long ne_tailleCarte = m_tmapSize;
	float* carteTop = m_tmap.data();

	/* Lecture des exemples */
	sigmamax *= a0;
	sigmamin *= a0;

	indexexe.resize(NbExempl);


	NeNormer(exemples, exempMax.data(), exempMin.data(), DimInput, NbExempl, typenorm);
	for(i=0; i < DimInput; i++) {
		qDebug() << " i " << i << " min " << exempMin[i] << " max " << exempMax[i];
	}

	for (i = 0; i < NbExempl; i++) {
		for (j = 0; j < DimInput ; j++) {
			ij=i*DimInput + j ;
			exemples[ij] = (exemples[ij] - exempMin[j]) /
						   (exempMax[j] - exempMin[j]) - 0.5 ;
		}
	}

	passigma = (sigmamax - sigmamin) /nbcycles;
	pasepsil = (epsilmax - epsilmin) / nbcycles;

	/* Les poids cellulaires sont initialises par tirage
	   au hasard d'un nombre compris entre -0.1 et +0.1 */
	srand(is);
	for(i = 0; i < tailltot; i++) {
		carteTop[i] = ((float) rand() - rand_min) / rand_max / 5.0 ;
	}

	/*******************************************************************/
	/********************* Phase d'apprentissage ***********************/
	/*******************************************************************/

	/* Initialisations */
	hist.resize(ne_tailleCarte);
	epsilon = epsilmax;

	NN = ((NbExempl * 3) / 6) * 2;

	sigma = sigmamax;

	NormeCarteTop = 0;
	for (i = 0; i < NbExempl; i ++) {
		indexexe[i] = i;
	}
	/* Boucle sur les cycles d'apprentissage */
	nbcycles = ne_nbCycles/cysup ;
	for (csup = 0 ; csup < cysup ; csup ++) {
		epsilmax = epsilmax/(csup + 1);
		pasepsil = (epsilmax - epsilmin) / ((float) nbcycles);
		epsilon = epsilmax ;
		for (cy = 0; cy < nbcycles; cy++) {
			qDebug() << " csup " << csup << " cy " << cy;
			for (i = 0; i < ne_tailleCarte; i++) {
				hist[i] = 0;
			}

			/* Boucle sur les exemples pour modifier les poids cellulaires */
			for (ex1 = 0; ex1 < NbExempl; ex1 ++) {
				ex=indexexe[ex1] ;
				travail = 1.0e30;

				/* Recherche de la cellule optimale pour l'exemple 'ex' */
				for (i = 0; i < ne_tailleCarte; i++) {
					/* Calcul de la distance du polong courant a la cellule courante */
					distance = 0.0;
					distance_E = 0.0 ;
					compteur_c = i * DimInput;
					compteur_e = ex * DimInput;

					for (j = 0; j < DimInput ; j++) {
						distance_E += (exemples[compteur_e +j]-carteTop [compteur_c+j])
									  * (exemples[compteur_e +j] - carteTop [compteur_c+j]) ;
					}
					distance = distance_E ;
					/* Selection de la cellule la plus proche */
					if (distance < travail) {
						ex_opt = ex;
						ic_opt = i;
						travail = distance;
					}
				}
				(hist[ic_opt]) ++;

				/* Reajustement des poids cellulaires de la cellule selectionnee */
				compteur_c = ic_opt * DimInput;
				compteur_e = ex_opt * DimInput;

				for (j =  0; j < DimInput ; j++) {
					carteTop[compteur_c + j] +=  epsilon *
												 (exemples[compteur_e +j] - carteTop[compteur_c + j]);
				}

				/* Et dans le voisinage de la cellule selectionnee le cas echeant */
				if (voisinag) {
					for (ic = 0; ic < ne_tailleCarte; ic++) {
						if (ic_opt - ic) {
							distance =  (ic_opt - ic) * (ic_opt - ic);
							epsilov = epsilon * (float) exp((double)(-distance/sigma));
							for (j = 0; j < DimInput ; j++) {
								ij = ic * DimInput ;
								carteTop[ij+j] += epsilov * (exemples[compteur_e+j] - carteTop[ij+j]);
							}
						}
					}
				}

			}

			epsilon -= pasepsil;
			sigma -= passigma ;

			/* Permutation des index des exemples */

			for (i = 0; i < NbExempl - 1; i++) {
				j = (long) ( (float) rand () / rand_max * (float) (NbExempl - i - 1) );
				ij = indexexe[i];
				indexexe[i] = indexexe[i + j];
				indexexe[i + j] = ij;
			}
			j = (long) ( (float) rand () / rand_max * (float) (NbExempl - 1) );
			ij = indexexe[NbExempl -1];
			indexexe[NbExempl -1] = indexexe[j];
			indexexe[j] = ij;
		} /* Fin de boucle cy */
	} /* Fin Boucle csup */

	/* reorganisation de la Carte */
	/* recherche de couple optimal */
	w= 0;
	for (ij = 0, i = 0; i < ne_tailleCarte - 1; ij += DimInput, i++) {
		for (distance = 0.0, k = 0; k < DimInput;  k++) {
			distance += (carteTop[ij + k] - carteTop[ij + k + DimInput]) *
						(carteTop[ij + k] - carteTop[ij + k + DimInput]);
		}
		std::printf(" distance %8.2f ", (float) std::sqrt(distance));
		if( w < (float) std::sqrt(distance) ) {
			w = (float) std::sqrt(distance) ;
			ik= 0;
		}
	}
	ij=0 ;
	for (distance = 0.0, k = 0; k < DimInput;  k++) {
		distance += (carteTop[k] - carteTop[(ne_tailleCarte - 1)*DimInput+ k]) *
					(carteTop[k] - carteTop[(ne_tailleCarte - 1)*DimInput+ k]) ;
	}
	std::printf(" distance %8.2f \n", (float) std::sqrt(distance));
	if( w > (float) std::sqrt(distance) ) {
		w = (float) std::sqrt(distance) ;
		//ik< 0;
	}
	for (i=0 ; i< ne_tailleCarte ; i++) {
		ind = (ik +i) % ne_tailleCarte ;
		for(k = 0; k < DimInput;  k++) {
			CarteTop_N[i*DimInput + k] = carteTop[ind*DimInput + k] ;
		}
	}
	for(i=0 ; i < tailltot ; i++) {
		carteTop[i]= CarteTop_N[i];
	}

	for (ij = 0, i = 0; i < ne_tailleCarte; i++) {
		for (j = 0; j < DimInput; j++) {
			carteTop[ij] = (carteTop[ij] + 0.5) *
						   (exempMax[j] - exempMin[j]) + exempMin[j];
			ij++;
		}
	}
}

long KohonenEngine::applyTmap(float* example, bool *ok) const {
	long label;
	(*ok) = false;

	double distance_Min = 10e32;

	for (long i = 0; i < m_tmapSize; i++) {
		/* Calcul de la distance du point courant a la cellule courante */
		double distance_E = 0.0 ;
		long compteur_c = i * m_exampleSize;
		for(int j = 0; j < m_exampleSize; j++) {
			// ???? not buf ?
			distance_E += (example[j]-  m_tmap[compteur_c+j])
						  * (example[j] - m_tmap[compteur_c+j]) ;
		}
		if(distance_Min > distance_E) {
			distance_Min = distance_E ;
			label = i ;
			(*ok) = true;
		}
	}
	return label;
}

void KohonenEngine::NeNormer(float *exemples, float *exempmax, float *exempmin,
	            long diminput, long nbexempl, long typenorm) {
	long    i, j, ij;
	float   exmax, exmin;

	switch (typenorm) {
			/* Normalisation par variable (relative) */
	case 1:
		for (i = 0; i < diminput; i++) {
				exempmax[i] =  exemples[i];
				exempmin[i] =  exemples[i];
				for (ij = i + diminput, j = 1; j < nbexempl ; ij += diminput, j++) if ( i < exemples[ij-i] + 1) {
						if (exemples[ij] > exempmax[i])
								exempmax[i] = exemples[ij];
						if (exemples[ij] < exempmin[i])
								exempmin[i] =  exemples[ij];
				}
		}
		for (i = 0; i < diminput; i++) {
				printf(" variable no %d vmax %8.3f vmin %8.3f \n",  /*    a modifier ?     */
					i, exempmax[i], exempmin[i]);
		}
		break;

		/* Normalisation g<8e>n<8e>rale (absolue) */
	case 2:
		for (i = 0; i < diminput; i++)  {
				exempmax[i] =  exemples[i];
				exempmin[i] =  exemples[i];
				for (ij = i + diminput, j = 1; j < nbexempl; ij += diminput, j++) if ( i < exemples[ij-i] + 1)  {
						if (exemples[ij] > exempmax[i])
								exempmax[i] =  exemples[ij];
						if (exemples[ij] < exempmin[i])
								exempmin[i] =  exemples[ij];

				}
		}
		exmax = exempmax[1] ;
		exmin = exempmin[1] ;
		for ( i = 2; i < diminput; i++) {
				if (exempmax[i] > exmax)
						exmax = exempmax[i];
				if (exempmin[i] < exmin)
						exmin = exempmin[i];
		}
		for (i = 1; i < diminput; i++) {
				exempmax[i] = exmax;
				exempmin[i] = exmin;
		}
		break;
	case 3:
		for (i = 0; i < diminput; i++)  {
				exempmax[i] =  exemples[i];
				exempmin[i] =  exemples[i];
				for (ij = i + diminput, j = 1; j < nbexempl; ij += diminput, j++) if ( i < exemples[ij-i] + 1)  {
						if (exemples[ij] > exempmax[i])
								exempmax[i] =  exemples[ij];
						if (exemples[ij] < exempmin[i])
								exempmin[i] =  exemples[ij];
				}
		}
		exmax = exempmax[1];
		exmin = exempmin[1] ;
		printf(" exmax %f exmin %f\n",exmax,exmin);
		for (i = 2; i < diminput; i++) {
				if (exempmax[i] > exmax)
						exmax = exempmax[i];
				if (exempmin[i] < exmin)
						exmin = exempmin[i];
		}
		if(exmax < -exmin) {
				exmax = -exmin ;
		}
		else {
				exmin = -exmax ;
		}
		for (i = 1; i < diminput; i++) {
				exempmax[i] = exmax;
				exempmin[i] = exmin;
		}
		printf(" exmax %f exmin %f\n",exmax,exmin);
		break;

		/* Pas de normalisation */
	default:
		for (i = 0; i < diminput; i++) {
				exempmax[i] = 1.0;
				exempmin[i] = 0;
		}
		break;
	}
}

