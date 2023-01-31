/*
 * CUDAImageMask.h
 *
 *  Created on: Jan 25, 2022
 *      Author: l1046262
 */

#ifndef SRC_SLICER_GRAPHICITEMSRGT_CUDAIMAGEMASK_H_
#define SRC_SLICER_GRAPHICITEMSRGT_CUDAIMAGEMASK_H_

#include "imageformats.h"
#include "cpuimagepaletteholder.h"
#include "cudaimagetexturemapper.h"
#include "GraphEditor_ItemInfo.h"
#include <QByteArray>
#include <QPainterPath>
#include "GraphEditor_EllipseShape.h"

#define MASK_TEXTURE_UNIT 10

class CUDAImageMask
{
public:
	QByteArray getArray()
	{
		return m_array;
	}

protected:

	void setMaskTexture(int width, int height , QByteArray array, const IGeorefImage *image, QObject *parent)
	{
		ImageFormats::QSampleType type =  ImageFormats::QSampleType::UINT8;

		CPUImagePaletteHolder* palette = new CPUImagePaletteHolder(width,height,type ,image) ;

		palette->updateTexture(array,false);

		if(texturemapper!=nullptr)
		{
			texturemapper->deleteLater();
		}
		texturemapper = new CUDAImageTextureMapper(palette,parent);
	};

	QByteArray applyMaskTexture(QGraphicsItem* parent, int image_width, int image_height, QPolygonF poly)
	{
		GraphEditor_ItemInfo* parentItem = dynamic_cast<GraphEditor_ItemInfo* >(parent);
		if (parentItem)
		{
			QVector<int> i_vec;
			QVector<int> j_vec;

			foreach (QPointF p, poly)
			{
				i_vec.push_back(p.y());
				j_vec.push_back(p.x());
			}

			int di_min = *std::min_element(i_vec.constBegin(), i_vec.constEnd());
			int dj_min = *std::min_element(j_vec.constBegin(), j_vec.constEnd());
			int di_max = *std::max_element(i_vec.constBegin(), i_vec.constEnd());
			int dj_max = *std::max_element(j_vec.constBegin(), j_vec.constEnd());

			QByteArray byteArray(image_height*image_width,0);

			QPainterPath path;
			path.addPolygon(poly);
			{
				for (int i=0; i<image_height; i+=1)
				{
					for (int j=0; j<image_width; j+=1)
					{
						QPointF point(j,i);
						if (path.contains(point) )
						{
							byteArray[i*image_width + j] = 255; // we want to display it
						}
						else
						{
							byteArray[i*image_width + j] = 0; // do not display it display it
						}
					}
				}
			}
			return byteArray;
		}
		else
		{
			QByteArray newByteArray(1,255);
			return newByteArray;
		}
	};

	CUDAImageTextureMapper* texturemapper = nullptr;
	bool m_ApplyMask = false;
	QByteArray m_array;
};



#endif /* SRC_SLICER_GRAPHICITEMSRGT_CUDAIMAGEMASK_H_ */
