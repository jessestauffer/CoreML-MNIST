//
//  ViewController.swift
//  CoreML-MNIST
//
//  Created by Jesse Stauffer on 10/23/18.
//  Copyright © 2018 Jesse Stauffer. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    @IBOutlet weak var drawingView: DrawingView!
    @IBOutlet weak var resultLabel: UILabel!
    
    let model = MNISTModel()
    let context = CIContext()
    var pixelBuffer : CVPixelBuffer?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        let attributes = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        CVPixelBufferCreate(kCFAllocatorDefault,
                            28,
                            28,
                            kCVPixelFormatType_OneComponent8,
                            attributes,
                            &pixelBuffer)
    }

    @IBAction func clearInput(_ sender: Any) {
        drawingView.lines = []
        drawingView.setNeedsDisplay()
        resultLabel.text = ""
    }
    
    @IBAction func detectInput(_ sender: Any) {
        let viewContext = drawingView.getViewContext()
        let cgImage = viewContext?.makeImage()
        let ciImage = CIImage(cgImage: cgImage!)
        context.render(ciImage, to: pixelBuffer!)
        
        // make a prediction
        let output = try? model.prediction(image: pixelBuffer!)
        
        // handle output
        resultLabel.text = output?.classLabel
    }
    
}

