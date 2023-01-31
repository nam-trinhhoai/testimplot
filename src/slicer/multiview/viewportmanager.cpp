#include "viewportmanager.h"
#include <Qt3DRender/QViewport>

ViewportManager::ViewportManager(QObject *parent)
    : QObject(parent)
    , m_previousViewMode(FreeViewMode)
    , m_currentViewMode(FreeViewMode)
    , m_mainViewport(nullptr)
    , m_maximizedViewport(nullptr)
    , m_freeSliceSubviewPortHeight(0.0f)
    , m_freeSliceSubviewPortWidth(0.0f)
    , m_mainViewportRectForMode({
                                    {FreeViewMode, QRectF(0.0f, 0.0f, 1.0f, 1.0f)},
                                    {FreeSliceIntersectionMode, QRectF(0.0f, 0.0f, 0.6f, 1.0f)},
                                    {View2DIntersectionMode, QRectF(0.0f, 0.0f, 0.6f, 1.0f)}
                                })
{
    QObject::connect(this, &ViewportManager::currentViewModeChanged,
                     this, &ViewportManager::restoreViewportLayout);
    QObject::connect(this, &ViewportManager::freeSliceSubviewPortWidthChanged,
                     this, &ViewportManager::restoreViewportLayout);
    QObject::connect(this, &ViewportManager::freeSliceSubviewPortHeightChanged,
                     this, &ViewportManager::restoreViewportLayout);
    QObject::connect(this, &ViewportManager::maximizedViewportChanged,
                     this, &ViewportManager::restoreViewportLayout);
}

ViewportManager::ViewMode ViewportManager::previousViewMode() const
{
    return m_previousViewMode;
}

ViewportManager::ViewMode ViewportManager::currentViewMode() const
{
    return m_currentViewMode;
}

void ViewportManager::setCurrentViewMode(ViewportManager::ViewMode mode)
{
    if (m_previousViewMode != m_currentViewMode) {
        m_previousViewMode = m_currentViewMode;
        emit previousViewModeChanged();
    }

    if (m_currentViewMode != mode) {
        m_currentViewMode = mode;
        emit currentViewModeChanged();
    }
}

Qt3DRender::QViewport *ViewportManager::mainViewport() const
{
    return m_mainViewport;
}

void ViewportManager::setMainViewport(Qt3DRender::QViewport *viewport)
{
    if (viewport == m_mainViewport)
        return;
    m_mainViewport = viewport;
    emit mainViewportChanged();
}

Qt3DRender::QViewport *ViewportManager::maximizedViewport() const
{
    return m_maximizedViewport;
}

void ViewportManager::setMaximizedViewport(Qt3DRender::QViewport *viewport)
{
    if (m_maximizedViewport == viewport)
        return;
    m_maximizedViewport = viewport;
    emit maximizedViewportChanged();
}

float ViewportManager::freeSliceSubviewPortHeight() const
{
    return m_freeSliceSubviewPortHeight;
}

void ViewportManager::setFreeSliceSubviewPortHeight(float height)
{
    if (qFuzzyCompare(m_freeSliceSubviewPortHeight, height))
        return;
    m_freeSliceSubviewPortHeight = height;
    emit freeSliceSubviewPortHeightChanged();
}

float ViewportManager::freeSliceSubviewPortWidth() const
{
    return m_freeSliceSubviewPortWidth;
}

void ViewportManager::setFreeSliceSubviewPortWidth(float width)
{
    if (qFuzzyCompare(m_freeSliceSubviewPortWidth, width))
        return;
    m_freeSliceSubviewPortWidth = width;
    emit freeSliceSubviewPortWidthChanged();
}


int ViewportManager::nbView3D() const
{
	return m_nbView3D;
}

void ViewportManager::setNbView3D(int nbview)
{
	if( m_nbView3D != nbview)
	{
		m_nbView3D = nbview;
		emit nbView3DChanged();
	}
}

bool ViewportManager::modeSplit3D() const
{
	return m_modeSplit3D;

}


void ViewportManager::setModeSplit3D(bool split)
{
	if(m_modeSplit3D != split)
	{
		m_modeSplit3D = split;
		emit modeSplit3DChanged();
	}
}

void ViewportManager::fusionViewport()
{

}

void ViewportManager::addViewport(Qt3DRender::QViewport *viewport, ViewportManager::ViewportPlaceHolder placeHolder)
{
	const QRectF vpARect = viewport->normalizedRect();

	//qDebug()<<placeHolder<<"addViewport : "<<placeHolder;
    m_viewportToPlaceHolder.insert(viewport, placeHolder);
    restoreViewportLayout();
}

void ViewportManager::addViewportMulti(ViewRenderMode mode, Qt3DRender::QViewport *viewport, ViewportManager::ViewportPlaceHolder placeHolder,bool modeSplit)
{
	//if(mode == ViewMode3D) m_nbView3D++;
	if(m_nbView3D <= m_maxNbView3D)
	{
		const QRectF vpARect = viewport->normalizedRect();

		//qDebug()<<vpARect.width()<<" , addViewportMulti : "<<placeHolder;
		m_viewportToPlaceHolder.insert(viewport, placeHolder);
		restoreViewportLayout();
	}
}

void ViewportManager::removeViewport(Qt3DRender::QViewport *viewport)
{

	qDebug()<<"remove viewport";
	m_viewportToPlaceHolder.remove(viewport);

	/*auto it = m_viewportToPlaceHolder.cbegin();
	const auto end = m_viewportToPlaceHolder.cend();

	while (it != end) {

	}*/


	restoreViewportLayout();
}


QRectF ViewportManager::computeViewport(int numView ,Qt3DRender::QViewport *viewport)
{
	QRectF vpRect = viewport->normalizedRect();

	switch(m_nbView3D)
	{
		case 1:
		{
			vpRect = QRectF(0.0f,0.0f, vpRect.width(), 1.0f);
			break;
		}
		case 2:
		{
			vpRect = QRectF((float)numView/(float)m_nbView3D,0.0f, (float)vpRect.width()/(float)m_nbView3D, 1.0f);
			break;
		}
		case 3:
		{
			vpRect = QRectF((float)numView/(float)m_nbView3D,0.0f, (float)vpRect.width()/(float)m_nbView3D, 1.0f);
			break;
		}
		case 4:
		{
			float posX = (numView/2)*0.5f;
			float posY = (numView%2)*0.5f;
			float dimX = (1.0f - m_freeSliceSubviewPortWidth)*0.5f;
			float dimY = 0.5f;
			vpRect = QRectF(posX,posY, dimX, dimY);
			break;
		}
		default:
		{
			break;
		}
	}
	return vpRect;
}

void ViewportManager::swapViewports(Qt3DRender::QViewport *a, Qt3DRender::QViewport *b)
{
    Q_ASSERT(m_viewportToPlaceHolder.contains(a) && m_viewportToPlaceHolder.contains(b));



    // Swap rects for the two viewports
    const QRectF vpARect = a->normalizedRect();
    const QRectF vpBRect = b->normalizedRect();
    a->setNormalizedRect(vpBRect);
    b->setNormalizedRect(vpARect);

    // Swap place holders for the two viewports
    const ViewportPlaceHolder vpAPlaceHolder = m_viewportToPlaceHolder.value(a);
    const ViewportPlaceHolder vpBPlaceHolder = m_viewportToPlaceHolder.value(b);
    m_viewportToPlaceHolder[a] = vpBPlaceHolder;
    m_viewportToPlaceHolder[b] = vpAPlaceHolder;
}

void ViewportManager::placeFreeSliceViewport(Qt3DRender::QViewport *vp,
                                             ViewportPlaceHolder placeHolder)
{
    QRectF vpRect = QRectF(0.0f, 0.0f, 1.0f, 1.0f);



    switch (placeHolder) {
        case ViewportManager::PlaceHolder_1: {
            // Large Main Viewport


        	float width = (1.0f - m_freeSliceSubviewPortWidth)/(float)m_nbView3D;
        	float height = 1;
        	if(m_nbView3D == 4)
        	{
        		width = (1.0f - m_freeSliceSubviewPortWidth)/2.0f;
        		height =0.5f;
        	}


            vpRect = QRectF(0.0f, 0.0f, width, height);
            break;
        }
        case ViewportManager::PlaceHolder_2: {
            // Side Top Most Free Slice Viewport
            vpRect = QRectF(1.0f - m_freeSliceSubviewPortWidth,
                            0.0f,
                            m_freeSliceSubviewPortWidth,
                            m_freeSliceSubviewPortHeight);
            break;
        }
        case ViewportManager::PlaceHolder_3: {
            // Side Centered Free Slice Viewport
            vpRect = QRectF(1.0f - m_freeSliceSubviewPortWidth,
                            m_freeSliceSubviewPortHeight,
                            m_freeSliceSubviewPortWidth,
                            m_freeSliceSubviewPortHeight);
            break;
        }
        case ViewportManager::PlaceHolder_4: {
            // Bottom Centered Free Slice Viewport
        	float width = (1.0f - m_freeSliceSubviewPortWidth)/(float)m_nbView3D;
        	float posX = (1.0f - m_freeSliceSubviewPortWidth)/(float)m_nbView3D;
        	float height = 1.0f;
        	if(m_nbView3D == 4)
			{
        		posX =(1.0f - m_freeSliceSubviewPortWidth)/2.0f;
				width = (1.0f - m_freeSliceSubviewPortWidth)/2.0f;
				height =0.5f;
			}

        	vpRect = QRectF(posX, 0.0f, width , height);
            //vpRect = QRectF(1.0f - m_freeSliceSubviewPortWidth,2.0f * m_freeSliceSubviewPortHeight,m_freeSliceSubviewPortWidth,m_freeSliceSubviewPortHeight);


            break;
        }
        case ViewportManager::PlaceHolder_5: {
                   // Bottom Centered Free Slice Viewport
			float width = (1.0f - m_freeSliceSubviewPortWidth)/(float)m_nbView3D;
			float posX = 2* ((1.0f - m_freeSliceSubviewPortWidth)/(float)m_nbView3D);
			float height = 1.0f;
			if(m_nbView3D == 4)
			{
				posX =0.0f;
				width = (1.0f - m_freeSliceSubviewPortWidth)/2.0f;
				height =0.5f;
			}

			vpRect = QRectF(posX, 0.0f, width , height);
			   //vpRect = QRectF(1.0f - m_freeSliceSubviewPortWidth,2.0f * m_freeSliceSubviewPortHeight,m_freeSliceSubviewPortWidth,m_freeSliceSubviewPortHeight);


			   break;
		   }
        case ViewportManager::PlaceHolder_6: {
			// Bottom Centered Free Slice Viewport

			float width = (1.0f - m_freeSliceSubviewPortWidth)/2.0f;
			float height = 0.5f;
			float posX = (1.0f - m_freeSliceSubviewPortWidth)/2.0f;//     2* ((1.0f - m_freeSliceSubviewPortWidth)/(float)m_nbView3D);
			float posY = 0.5f;

			vpRect = QRectF(posX, posY, width , height);
			   //vpRect = QRectF(1.0f - m_freeSliceSubviewPortWidth,2.0f * m_freeSliceSubviewPortHeight,m_freeSliceSubviewPortWidth,m_freeSliceSubviewPortHeight);
			   break;
		   }
        default:
            break;
        }

 /*   switch (placeHolder) {
    case ViewportManager::PlaceHolder_0: {
          // Large Main Viewport
          vpRect =  QRectF(0.0f, 0.0f, 1.0f - m_freeSliceSubviewPortWidth, 0.5f);
          break;
      }
    case ViewportManager::PlaceHolder_1: {
        // Large Main Viewport
    	//vpRect = computeViewport(m_nbView3D,vp);//
        vpRect = QRectF(1.0f - m_freeSliceSubviewPortWidth, 0.5,  m_freeSliceSubviewPortWidth, m_freeSliceSubviewPortHeight);
        break;
    }
    case ViewportManager::PlaceHolder_2: {
        // Side Top Most Free Slice Viewport
        vpRect = QRectF(1.0f - m_freeSliceSubviewPortWidth,
                        0.0f,
                        m_freeSliceSubviewPortWidth,
                        m_freeSliceSubviewPortHeight);
        break;
    }
    case ViewportManager::PlaceHolder_3: {
           // Large Main Viewport
       	vpRect = computeViewport(m_nbView3D,vp);//
           //vpRect = QRectF(0.0f, 0.5, 1.0f - m_freeSliceSubviewPortWidth, 0.5f);
           break;
       }


    case ViewportManager::PlaceHolder_4: {
        // Side Centered Free Slice Viewport
        vpRect = QRectF(1.0f - m_freeSliceSubviewPortWidth,
                        m_freeSliceSubviewPortHeight,
                        m_freeSliceSubviewPortWidth,
                        m_freeSliceSubviewPortHeight);
        break;
    }
    case ViewportManager::PlaceHolder_5: {
        // Bottom Centered Free Slice Viewport
        vpRect = QRectF(1.0f - m_freeSliceSubviewPortWidth,
                        2.0f * m_freeSliceSubviewPortHeight,
                        m_freeSliceSubviewPortWidth,
                        m_freeSliceSubviewPortHeight);
        break;
    }
    default:
        break;
    }
*/
  //  qDebug()<<"placeFreeSliceViewport : "<<placeHolder;
    vp->setNormalizedRect(vpRect);
}


void ViewportManager::restoreViewportLayout()
{

    // Given the current mode, place viewports according to their PlaceHolder
    // In the free slice mode, the main viewport may actually be in the
    // FreeSlice sub viewports
    if (m_currentViewMode == FreeSliceIntersectionMode) {
        auto it = m_viewportToPlaceHolder.cbegin();
        const auto end = m_viewportToPlaceHolder.cend();

        while (it != end) {
            Qt3DRender::QViewport *vp = it.key();
            const ViewportPlaceHolder placeHolder = it.value();
            placeFreeSliceViewport(vp, placeHolder);

            // Handle maximized viewport case
            // Disable the non maximized viewports
            vp->setEnabled(m_maximizedViewport == nullptr || m_maximizedViewport == vp);
            // Make the maximized viewport take up all space
            if (m_maximizedViewport == vp)
                vp->setNormalizedRect(QRectF(0.0f, 0.0f, 1.0f, 1.0f));
            ++it;
        }
    } else {
        // Restore the mainViewport rect based on the view mode
        if (m_mainViewport != nullptr) {
            const QRectF currentViewportRect = m_mainViewport->normalizedRect();
            m_mainViewportRectForMode[m_previousViewMode] = currentViewportRect;
            const QRectF newViewportRect = m_mainViewportRectForMode[m_currentViewMode];
            m_mainViewport->setNormalizedRect(newViewportRect);
        }

        // Only enable the main viewport
        auto it = m_viewportToPlaceHolder.cbegin();
        const auto end = m_viewportToPlaceHolder.cend();

        while (it != end) {
            Qt3DRender::QViewport *vp = it.key();
            vp->setEnabled(m_mainViewport == vp);
            ++it;
        }
    }
}
