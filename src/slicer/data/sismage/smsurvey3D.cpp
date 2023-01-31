#include "smsurvey3D.h"
#include "Xt.h"
#include <QVector>
#include <QPoint>
#include <QFileInfo>
#include "sismagedbmanager.h"
#include "smtopo3ddesc.h"
#include "seautils.h"
#include "SeismicManager.h"
#include <iomanip>

#include <gdal.h>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace boost::numeric::ublas;

SmSurvey3D::SmSurvey3D(const std::string &surveyPath) {
	m_surveyPath = surveyPath;
	std::string gemetryDesc = surveyPath + "/DATA/SEISMIC/geometry.desc";
	int result = SeismicManager::filextGetAxis(QString::fromStdString(gemetryDesc));
	if (result!=2) { // 2 is the error number
		inri::Xt xt(gemetryDesc);

		std::string topo3dDescPath = surveyPath + "/DATA/TOPOS/topo3d.desc";
		SmTopo3dDesc topo3dDesc(topo3dDescPath);
		if (topo3dDesc.isValid()) {
			m_firstInline = xt.startSlice();
			m_inlineDim = xt.nSlices();
			m_inlineStep = topo3dDesc.getInlineStep();
			m_inlineDist = topo3dDesc.getInlineDist();
			m_firstXline = xt.startRecord();
			m_xlineDim = xt.nRecords();
			m_xlineStep = topo3dDesc.getCrosslineStep();
			m_xlineDist = topo3dDesc.getCrosslineDist();

			m_valid = true;
		}
	} else {
		std::cerr<<"SmSurvey3D bad geometry.desc file : "<<gemetryDesc<<std::endl;
	}

	m_inlineToXYTransfo = nullptr;
	m_ijToXYTransfo = nullptr;
}

SmSurvey3D::~SmSurvey3D() {
	if (m_inlineToXYTransfo)
		delete m_inlineToXYTransfo;
	if (m_ijToXYTransfo)
		delete m_ijToXYTransfo;
}

void SmSurvey3D::computeTransformations() {
	if (m_inlineToXYTransfo != nullptr)
		return;

	std::string topoPath = SismageDBManager::topoExchangePathFromSurveyPath(
			m_surveyPath);

	if (!QFileInfo(QString::fromStdString(topoPath)).exists()) {
		SeaUtils::createSurveyTopoFile(QString::fromStdString(m_surveyPath));
	}

	QVector<QPointF> ijPoints;
	QVector<QPointF> ilXlPoints;
	QVector<QPointF> xyPoints;
	std::ifstream myfile(topoPath.c_str());
	if (myfile.is_open()) {
		std::string line;
		while (std::getline(myfile, line)) {
			if (line.empty() || line.find('#') == 0)
				continue;
			std::istringstream iss(line);
			std::array<double, 6> col;
			if (!(iss >> col[0] >> col[1] >> col[2] >> col[3] >> col[4]
					>> col[5])) {
				std::cerr << "Invalid topo description file" << std::endl;
			}

			ijPoints.push_back(QPointF(col[0], col[1]));
			ilXlPoints.push_back(QPointF(col[2], col[3]));
			xyPoints.push_back(QPointF(col[4], col[5]));
		}
		myfile.close();

		GDAL_GCP gcpIJ[ijPoints.size()];
		GDAL_GCP gcpILXL[ijPoints.size()];
		char const *voidChar = "";
		for (int i = 0; i < ijPoints.size(); i++) {
			gcpIJ[i] = GDAL_GCP { (char*) voidChar, (char*) voidChar,
					ijPoints[i].y(), ijPoints[i].x(), xyPoints[i].x(),
					xyPoints[i].y(), 0 };
			gcpILXL[i] = GDAL_GCP { (char*) voidChar, (char*) voidChar,
					ilXlPoints[i].y(), ilXlPoints[i].x(), xyPoints[i].x(),
					xyPoints[i].y(), 0 };
		}
		double geotransformIJ[6];
		double geotransformIlXl[6];
		int retVal = GDALGCPsToGeoTransform(ijPoints.size(), gcpIJ, geotransformIJ,
				false);
		if (retVal == 0)
			std::cerr << "Failed to find a correct Geotransform" << std::endl;

		retVal = GDALGCPsToGeoTransform(ijPoints.size(), gcpILXL, geotransformIlXl,
				false);
		if (retVal == 0)
			std::cerr << "Failed to find a correct Geotransform" << std::endl;

		std::array<double, 6> resultIJ;
		std::array<double, 6> resultILXL;
		for (int i = 0; i < 6; i++) {
			resultIJ[i] = geotransformIJ[i];
			resultILXL[i] = geotransformIlXl[i];
		}

		m_inlineToXYTransfo = new Affine2DTransformation(m_xlineStep * m_xlineDim,
				m_inlineStep * m_inlineDim, resultILXL);
		m_ijToXYTransfo = new Affine2DTransformation(m_xlineDim, m_inlineDim,
				resultIJ);
	}else
	{
		std::cerr<<topoPath<<" NOT FOUND! Positionning is wrong!!!!!"<<std::endl;


		std::array<double, 6> result;
		result[0]=m_firstXline;
		result[1]=m_xlineStep;
		result[2]=0;

		result[3]=m_firstInline;
		result[4]=0;
		result[5]=m_inlineStep;
		m_inlineToXYTransfo = new Affine2DTransformation(m_xlineDim,m_inlineDim);
		m_ijToXYTransfo = new Affine2DTransformation(m_xlineDim, m_inlineDim,result);
	}



//		double x,y,x1,y1;
//		m_ijToXYTransfo->imageToWorld(0,0,x,y);
//		m_inlineToXYTransfo->imageToWorld(m_firstInline,m_firstXline,x1,y1);
//		std::cout<<"["<<x<<","<<y<<"],["<<x1<<","<<y1<<std::endl;

//Compute an transformation
//https://stackoverflow.com/questions/11687281/transformation-between-two-set-of-points
	//	matrix<double> A(xyPoints.size()*2, 4);
	//	matrix<double> y(xyPoints.size()*2, 1);
	//	for (int i = 0; i < xyPoints.size(); i++) {
	//		QPointF p = ijPoints[i];
	//		QPointF q = xyPoints[i];
	//
	//		A(2 * i, 0) = p.x();
	//		A(2 * i, 1) = p.y();
	//		A(2 * i, 2) = 1;
	//		A(2 * i, 3) = 0;
	//		A(2 * i + 1, 0) = p.y();
	//		A(2 * i + 1, 1) = -p.x();
	//		A(2 * i + 1, 2) = 0;
	//		A(2 * i + 1, 3) = 1;
	//
	//		y(2 * i, 0) = q.x();
	//		y(2 * i + 1, 0) = q.y();
	//	}
	//	std::cout <<std::fixed << std::setprecision(2)<< A << std::endl;
	//	std::cout <<std::fixed << std::setprecision(2)<< y << std::endl;
	//
	//	matrix<double> C = prod(trans(A), A);
	//	matrix<double> CInv = identity_matrix<float>(C.size1());
	//	permutation_matrix<size_t> pm(C.size1());
	//	lu_factorize(C, pm);
	//	lu_substitute(C, pm, CInv);
	//
	//	matrix<double> Ay = prod(trans(A), y);
	//	matrix<double> transfo = prod(CInv, Ay);
	//	std::cout << transfo << std::endl;
	//
	//	std::array<double, 6> result;
	//
	//	result[0]=transfo(2,0);
	//	result[1]=transfo(0,0);
	//	result[2]=transfo(1,0);
	//
	//	result[3]=transfo(3,0);
	//	result[4]=-transfo(1,0);
	//	result[5]=transfo(0,0);

}

Affine2DTransformation SmSurvey3D::inlineXlineToXYTransfo() {
	computeTransformations();
	return *m_inlineToXYTransfo;
}

Affine2DTransformation SmSurvey3D::ijToXYTransfo() {
	computeTransformations();
	return *m_ijToXYTransfo;
}

bool SmSurvey3D::isValid() const {
	return m_valid;
}

//
//bool SmSurvey3D::checkTransformation()
//{
//	std::array<double, 8> corners={0,0,m_inlineDim-1,0,0,m_xlineDim-1,m_inlineDim-1,m_xlineDim-1};
//	Affine2DTransformation tXY = ijToXYTransfo();
//	Affine2DTransformation tILXL = ijToILXLTransfo();
//	double x, y;
//	double il,xl;
//	for(int i=0;i<4;i++)
//	{
//		tXY.imageToWorld(corners[2*i],corners[2*i+1],x,y);
//		tILXL.imageToWorld(corners[2*i],corners[2*i+1],il,xl);
//		std::cout<<corners[2*i]<<"\t"<< corners[2*i+1]<<std::fixed << std::setprecision(2)<<"\t"<<x<<"\t"<<y<<"\t"<<il<<"\t"<<xl<<std::endl;
//	}
//
//	return true;
//}

