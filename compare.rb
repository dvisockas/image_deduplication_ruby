require "torch"
require "torchvision"
require "numo/narray"

module NeuralNet
  module_function
  def model
    @@model ||= begin
      net = TorchVision::Models::ResNet18.new
      net.load_state_dict(Torch.load("net.pth"))
      # Removing last layer (fully connected) to get embeddings instead of imagenet class probablities
      Torch::NN::Sequential.new(*net.children[...-1])
    end
  end

  def infer(input)
    model.call(transforms.call(input).unsqueeze(0))
  end

  def transforms
    TorchVision::Transforms::Compose.new([
      # TorchVision::Transforms::Resize.new(256),
      # TorchVision::Transforms::CenterCrop.new(224),
      TorchVision::Transforms::ToTensor.new,
      # TorchVision::Transforms::Normalize.new([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  end
end

class Image < Struct.new(:source_path)
  def embeddings
    Numo::SFloat.cast(
      NeuralNet
        .infer(Vips::Image.new_from_file(@source_path))
        .flatten
        .to_a
    )
  end

  def similarity(image)
    Math.sqrt((embeddings - image.embeddings).square.sum)
  end

  def torch_input
    transforms.call(Vips::Image.new_from_file(source_path)).unsqueeze(0)
  end
end

module ImageSimilarityPairs
  module_function
  def call(image_sources)
    pairs = {}
    image_sources.each do |column_source|
      image_sources.each do |row_source|
        key = [column_source, row_source].sort
        next if pairs[key] || column_source == row_source

        pairs[key] = Image.new(column_source).similarity(Image.new(row_source))
      end
    end
    pairs
  end
end

simlarity_pairs = ImageSimilarityPairs.call([
  "images/target.webp",
  "images/target.jpg",
  "images/target_smaller.webp",
  "images/target_different.webp",
  "images/target_way_different.webp",
])

simlarity_pairs.each do |(image, another_image), difference|
  puts "#{image} and #{another_image} similarity score: #{1 - difference.round(2)}"
end
