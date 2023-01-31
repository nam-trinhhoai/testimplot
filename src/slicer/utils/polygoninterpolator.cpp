#include "polygoninterpolator.h"

#include <cmath>

std::pair<Deviations, std::vector<std::size_t>> polygonInterpolator(
		const Deviations& deviation, double thresholdDeviation) {
	Deviations dvs;
	std::size_t ND = deviation.xs.size();

	if (ND==0) {
		return std::pair<Deviations, std::vector<std::size_t>>(deviation, std::vector<std::size_t>());
	}

	dvs.xs.push_back(deviation.xs[0]);
	dvs.xs.push_back(deviation.xs[ND-1]);
	dvs.ys.push_back(deviation.ys[0]);
	dvs.ys.push_back(deviation.ys[ND-1]);
	dvs.tvds.push_back(deviation.tvds[0]);
	dvs.tvds.push_back(deviation.tvds[ND-1]);
	dvs.mds.push_back(deviation.mds[0]);
	dvs.mds.push_back(deviation.mds[ND-1]);
	dvs.appliedDatum = deviation.appliedDatum;

	std::vector<std::size_t> dvsIndexes;
	dvsIndexes.push_back(0);
	dvsIndexes.push_back(ND-1);

	size_t idxDvs = 0 ;
	while (idxDvs<dvs.xs.size()-1) {
		bool newPointCreated = true;
		while(newPointCreated) {
			newPointCreated = false;
			double maxdist = 0.0;
			std::size_t indMaxdist;

			double x1=dvs.xs.at(idxDvs);
			double x2=dvs.xs.at(idxDvs+1);
			double y1=dvs.ys.at(idxDvs);
			double y2=dvs.ys.at(idxDvs+1);
			double z1=dvs.tvds.at(idxDvs);
			double z2=dvs.tvds.at(idxDvs+1);
			QVector3D p1(x1, y1, z1);
			QVector3D p2(x2, y2, z2);
//			it = DVS.begin()+idx_dvs ;
			for(int i=dvsIndexes[idxDvs]+1; i< dvsIndexes[idxDvs+1] ; i++) {
				QVector3D p0(deviation.xs.at(i), deviation.ys.at(i), deviation.tvds.at(i));
				double dist = distPointSegment(p0,p1,p2) ;
				if(dist > maxdist) {
					maxdist = dist ;
					indMaxdist = i ;
				}
			}
			if(maxdist > thresholdDeviation) {
				double x = deviation.xs.at(indMaxdist);
				double y = deviation.ys.at(indMaxdist);
				double z = deviation.tvds.at(indMaxdist);
				double md = deviation.mds.at(indMaxdist);
				dvs.xs.insert(dvs.xs.begin() + idxDvs + 1, x);
				dvs.ys.insert(dvs.ys.begin() + idxDvs + 1, y);
				dvs.tvds.insert(dvs.tvds.begin() + idxDvs + 1, z);
				dvs.mds.insert(dvs.mds.begin()+ idxDvs+1, md);

				dvsIndexes.insert(dvsIndexes.begin()+idxDvs+1, indMaxdist);

				newPointCreated = true ;
			}
		}
		idxDvs++;
	}
	return std::pair<Deviations, std::vector<std::size_t>>(dvs, dvsIndexes);
}

double scalarProduct(QVector3D p0, QVector3D p1, QVector3D p2) {
	double v ;
	v = (p1.x() - p0.x())*(p2.x() - p1.x()) +
		(p1.y() - p0.y())*(p2.y() - p1.y()) +
		(p1.z() - p0.z())*(p2.z() - p1.z()) ;
	return(v*v) ;
}

double distPointSegment (QVector3D p0, QVector3D p1, QVector3D p2) {
	double v ;
	v=(segmentLength(p1,p0) * segmentLength(p2,p1) - scalarProduct(p0,p1,p2)) / segmentLength(p1,p2) ;
	v = std::sqrt(v) ;
	return(v) ;
}
double segmentLength(QVector3D p1, QVector3D p2)
{
	return ( (p1.x() - p2.x())*(p1.x() - p2.x()) +
		(p1.y() - p2.y())*(p1.y() - p2.y()) +
		(p1.z() - p2.z())*(p1.z() - p2.z()) ) ;
}

