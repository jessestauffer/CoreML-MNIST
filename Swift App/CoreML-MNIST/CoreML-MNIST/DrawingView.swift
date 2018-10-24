//
//  DrawingView.swift
//  CoreML-MNIST
//
//  Created by Jesse Stauffer on 10/23/18.
//  Copyright © 2018 Jesse Stauffer. All rights reserved.
//

import UIKit

class DrawingView: UIView {

    var lineWidth = CGFloat(15)
    var color = UIColor.white
    
    var lines : [Line] = []
    var lastPoint : CGPoint!
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        lastPoint = touches.first!.location(in: self)
    }
    
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        let newPoint = touches.first!.location(in: self)
        lines.append(Line(startingPoint: lastPoint, endingPoint: newPoint))
        lastPoint = newPoint
        
        // make call to draw
        setNeedsDisplay()
    }
    
    override func draw(_ rect: CGRect) {
        super.draw(rect)
        
        let drawPath = UIBezierPath()
        drawPath.lineCapStyle = .round
        
        for line in lines {
            drawPath.move(to: line.startingPoint)
            drawPath.addLine(to: line.endingPoint)
        }
        
        drawPath.lineWidth = lineWidth
        color.set()
        drawPath.stroke()
    }
    
    func getViewContext() -> CGContext? {
        let colorSpace:CGColorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGImageAlphaInfo.none.rawValue
        let context = CGContext(data: nil, width: 28, height: 28, bitsPerComponent: 8, bytesPerRow: 28, space: colorSpace, bitmapInfo: bitmapInfo)
        context!.translateBy(x: 0 , y: 28)
        context!.scaleBy(x: 28/self.frame.size.width, y: -28/self.frame.size.height)
        self.layer.render(in: context!)
        
        return context
    }

}

class Line {
    var startingPoint : CGPoint
    var endingPoint : CGPoint
    
    init(startingPoint: CGPoint, endingPoint: CGPoint) {
        self.startingPoint = startingPoint
        self.endingPoint = endingPoint
    }
}
