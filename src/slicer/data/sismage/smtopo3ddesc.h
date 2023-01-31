/*
 * SmTopo3dDesc.h
 *
 *  Created on: Apr 8, 2020
 *      Author: l0222891
 */

#ifndef TARUMAPP_SRC_SMTOPO3DDESC_H_
#define TARUMAPP_SRC_SMTOPO3DDESC_H_


#include <string>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <algorithm>

class SmTopo3dDesc {

public:

	SmTopo3dDesc( const std::string& filename) {
		m_valid = readDesc( filename );
		if (!m_valid) {
			std::cerr<<"SmTopo3dDesc bad file : "<<filename<<std::endl;
		}
	}

	bool readDescFromSeismic(const std::string& filename) {
		size_t found = filename.find_last_of("/\\");
		const std::string folder = filename.substr(0, found - 1);
		const std::string topo3dFile = folder + "../TOPO/topo3d.desc";

		//struct stat buffer;
		if ( access ( topo3dFile.c_str(), F_OK) != -1)
			readDesc(topo3dFile);
		else
			return false;
	}

	bool readDesc(const std::string& filename) {

		std::string _ObjectType;
		unsigned _NDims;
		long _HeaderSize;
		std::string _DimSize;
		std::string _ElementType;
		std::string _ElementDataFile;
		std::string line;
		std::ifstream myfile( filename.c_str());
		bool ok = myfile.is_open();
		if (ok) {
			while ( std::getline (myfile,line) ) {
				int len = line.length();
				const size_t b = line.find_last_of("=");
				const std::string nameStr = line.substr(0, b);
				const std::string valueStr = line.substr(b+1, len-1);
				if ( nameStr.compare("topo_3d_crossline_angle") == 0) {
					m_crosslineAngle = (double) atof(valueStr.c_str());
				}
				else if ( nameStr.compare("topo_3d_inline_angle") == 0) {
					m_inlineAngle = (double) atof(valueStr.c_str());
				}
				else if ( nameStr.compare("topo_3d_x_origin") == 0) {
					m_xOrigin = (double) atof(valueStr.c_str());
				}
				else if ( nameStr.compare("topo_3d_y_origin") == 0) {
					m_yOrigin = (double) atof(valueStr.c_str());
				}
				else if ( nameStr.compare("topo_3d_first_inline_number") == 0) {
					m_firstInlineNumber = (double) atof(valueStr.c_str());
				}
				else if ( nameStr.compare("topo_3d_first_cross_line_number") == 0) {
					m_firstCrossLineNumber = (double) atof(valueStr.c_str());
				}
				else if ( nameStr.compare("topo_3d_inline_step") == 0) {
					m_inlineStep = (double) atof(valueStr.c_str());
				}
				else if ( nameStr.compare("topo_3d_inline_dist") == 0) {
					m_inlineDist = (double) atof(valueStr.c_str());
				}
				else if ( nameStr.compare("topo_3d_crossline_step") == 0) {
					m_crosslineStep = (double) atof(valueStr.c_str());
				}
				else if ( nameStr.compare("topo_3d_crossline_dist") == 0) {
					m_crosslineDist = (double) atof(valueStr.c_str());
				}
			  }
			myfile.close();
		}
		//else throw io::CubeIOException(std::string("Unable to open file ") + filename.c_str());

		return ok;
	}

	double getCrosslineAngle() const {
		return m_crosslineAngle;
	}

	double getCrosslineDist() const {
		return m_crosslineDist;
	}

	double getCrosslineStep() const {
		return m_crosslineStep;
	}

	double getFirstCrossLineNumber() const {
		return m_firstCrossLineNumber;
	}

	double getFirstInlineNumber() const {
		return m_firstInlineNumber;
	}

	double getInlineAngle() const {
		return m_inlineAngle;
	}

	double getInlineDist() const {
		return m_inlineDist;
	}

	double getInlineStep() const {
		return m_inlineStep;
	}

	double getXOrigin() const {
		return m_xOrigin;
	}

	double getYOrigin() const {
		return m_yOrigin;
	}

	bool isValid() const {
		return m_valid;
	}

private:
	double m_crosslineAngle=2.4755;
	double m_inlineAngle=0.9047036732;
	double m_xOrigin=763985;
	double m_yOrigin=9400454;
	double m_firstInlineNumber=484;
	double m_firstCrossLineNumber=3773;
	double m_inlineStep=2;
	double m_inlineDist=25;
	double m_crosslineStep=2;
	double m_crosslineDist=25;
	bool m_valid = false;
};


#endif /* TARUMAPP_SRC_SMTOPO3DDESC_H_ */
